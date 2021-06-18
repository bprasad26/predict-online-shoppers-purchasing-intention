import os
from dash_core_components.Dropdown import Dropdown
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from dash_bootstrap_components._components.CardHeader import CardHeader
from dash_bootstrap_components._components.FormText import FormText
from dash_html_components.Br import Br
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import shap
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
import base64
import pydot
import xml
from data_module import prepare_data
import json
import joblib
import warnings

warnings.filterwarnings("ignore")
# warnings.simplefilter(action="ignore", category=UserWarning)
# warnings.simplefilter(action="ignore", category=DeprecationWarning)

# project paths
data_path = os.path.join(os.getcwd(), "data")
model_path = os.path.join(os.getcwd(), "models")
json_col_path = os.path.join(model_path, "columns.json")

# read data
train = pd.read_csv(os.path.join(data_path, "Train.csv"))
test = pd.read_csv(os.path.join(data_path, "Test.csv"))

# prepare data
X_train, y_train, X_test, full_pipe = prepare_data(train, test)

num_cols = X_train.select_dtypes(exclude="object").columns.tolist()
cat_cols = X_train.select_dtypes(include="object").columns.tolist()

X_train = full_pipe.fit_transform(X_train)
X_test = full_pipe.transform(X_test)


# get the list of one-hot encoded categories
ohe_categories = full_pipe.named_transformers_.cat.named_steps.onehotencoder.categories_
# create ohe-hot encoded category column names
new_ohe_features = [
    f"{col}__{val}" for col, vals in zip(cat_cols, ohe_categories) for val in vals
]

# list of all features names
all_features = num_cols + new_ohe_features

X_train = pd.DataFrame(X_train, columns=all_features)
X_test = pd.DataFrame(X_test, columns=all_features)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

# reset index
X_valid = X_valid.reset_index().drop("index", axis=1)
y_valid = y_valid.reset_index().drop("id", axis=1)

# convert from seconds to minutes
X_train["ProductRelated_Duration"] = X_train["ProductRelated_Duration"] / 60
X_valid["ProductRelated_Duration"] = X_valid["ProductRelated_Duration"] / 60
X_test["ProductRelated_Duration"] = X_test["ProductRelated_Duration"] / 60

# month columns
month_cols = X_train.filter(like="Month").columns.tolist()
# visitor type columns
visitor_type_cols = X_train.filter(like="VisitorType").columns.tolist()

# select only important features that was used in the model
json_col_path = os.path.join(os.getcwd(), "models", "columns.json")
with open(json_col_path, "r") as j:
    contents = json.loads(j.read())

cols_to_use = contents["data_columns"]
X_train_sel = X_train[cols_to_use].copy()
X_valid_sel = X_valid[cols_to_use].copy()
X_test_sel = X_test[cols_to_use].copy()


# dcc.Markdown("##### Website - [Life With Data](https://www.lifewithdata.com/)"),
# dcc.Markdown(
#     "##### Code - [GitHub](https://github.com/bprasad26/online_shoppers_purchasing_intension_prediction)"


controls = dbc.Card(
    [
        # html.H4("Model Input"),
        dbc.CardHeader(
            "Model Input",
            className="bg-primary text-white",
        ),
        html.Br(style={"margin-bottom": "20px"}),
        dbc.FormGroup(
            [
                dbc.Label("Month", size="md"),
                dcc.Dropdown(
                    id="month-selector",
                    options=[{"label": col, "value": col} for col in month_cols],
                    value="Month__Sep",
                ),
            ],
        ),
        dbc.FormGroup(
            [
                dbc.Label("Visitor Type", size="md"),
                dcc.Dropdown(
                    id="visitor-type-selector",
                    options=[{"label": col, "value": col} for col in visitor_type_cols],
                    value="VisitorType__New_Visitor",
                ),
            ],
        ),
        dbc.FormGroup(
            [
                dbc.Label("Page Value", size="md"),
                dbc.Input(
                    id="page-value-input",
                    value="37",
                    type="text",  # passing numbers causing unexpected behaviour
                    debounce=True,
                ),
                dbc.FormText("Enter a value between $0 to $300.", color="secondary"),
            ],
        ),
        dbc.FormGroup(
            [
                dbc.Label("Exit Rate", size="md"),
                dbc.Input(
                    id="exit-rate-input",
                    value="0.03",
                    type="text",  # passing numbers causing unexpected behaviour
                    debounce=True,
                ),
                dbc.FormText("Enter a value between 0 and 1"),
                # dcc.Slider(
                #     id="exit-rate-slider",
                #     min=0.0,
                #     max=1.0,
                #     step=0.01,
                #     value=0.05,
                #     marks={
                #         0: "0",
                #         0.1: "0.1",
                #         0.3: "0.3",
                #         0.5: "0.5",
                #         0.7: "0.7",
                #         0.9: "0.9",
                #         1: "1",
                #     },
                # ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Bounce Rate", size="md"),
                dbc.Input(
                    id="bounce-rate-input",
                    value="0",
                    type="text",  # passing numbers causing unexpected behaviour
                    debounce=True,
                ),
                dbc.FormText("Enter a value between 0 and 1")
                # dcc.Slider(
                #     id="bounce-rate-slider",
                #     min=0.0,
                #     max=1.0,
                #     step=0.01,
                #     value=0.05,
                #     marks={
                #         0: "0",
                #         0.1: "0.1",
                #         0.3: "0.3",
                #         0.5: "0.5",
                #         0.7: "0.7",
                #         0.9: "0.9",
                #         1: "1",
                #     },
                # ),
            ],
        ),
        dbc.FormGroup(
            [
                dbc.Label("Product Related", size="md"),
                dbc.Input(
                    id="product-related-input", value="34", type="text", debounce=True
                ),
                # dcc.Slider(
                #     id="product-related-slider",
                #     min=0,
                #     max=200,
                #     step=1,
                #     value=18,
                #     marks={
                #         0: "0",
                #         30: "30",
                #         60: "60",
                #         90: "90",
                #         150: "150",
                #         200: "200",
                #     },
                # ),
                dbc.FormText(
                    "Number of times user visited product related section (0-200).",
                    color="secondary",
                ),
            ],
        ),
        dbc.FormGroup(
            [
                dbc.Label("Product Related Duration", size="md"),
                dbc.Input(
                    id="product-related-duration-input",
                    value="80",
                    type="text",
                    debounce=True,
                ),
                # dcc.Slider(
                #     id="product-related-duration-slider",
                #     min=0,
                #     max=100,
                #     step=1,
                #     value=5,
                #     marks={
                #         0: "0",
                #         5: "5",
                #         10: "10",
                #         20: "20",
                #         30: "30",
                #         60: "60",
                #         100: "100",
                #     },
                # ),
                dbc.FormText(
                    "Total Time spent in product related section (0-100 min.)"
                ),
            ],
        ),
    ],
    body=True,
    # color="primary",
    # inverse=True,
)

class_output_card = dbc.Card(
    [
        dbc.CardHeader("Real-time Prediction", className="bg-primary text-white"),
        html.P(
            "Class 1: customer will buy, Class 0: will not buy",
            style={"text-align": "center"},
        ),
        dbc.CardBody(html.H2(id="predicted-class", style={"text-align": "center"})),
    ],
    style={"width": "25rem"},
    # color="success",
    className="mx-auto",
)

###### plot Decision tree


def svg_to_fig(svg_bytes, title=None, plot_bgcolor="white", x_lock=False, y_lock=True):
    svg_enc = base64.b64encode(svg_bytes)
    svg = f"data:image/svg+xml;base64, {svg_enc.decode()}"

    # Get the width and height
    xml_tree = xml.etree.ElementTree.fromstring(svg_bytes.decode())
    img_width = int(xml_tree.attrib["width"].strip("pt"))
    img_height = int(xml_tree.attrib["height"].strip("pt"))

    fig = go.Figure()
    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width],
            y=[img_height, 0],
            mode="markers",
            marker_opacity=0,
            hoverinfo="none",
        )
    )
    fig.add_layout_image(
        dict(
            source=svg,
            x=0,
            y=0,
            xref="x",
            yref="y",
            sizex=img_width,
            sizey=img_height,
            opacity=1,
            layer="below",
        )
    )

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(showgrid=False, visible=False, range=[0, img_width])
    fig.update_yaxes(showgrid=False, visible=False, range=[img_height, 0])

    if x_lock is True:
        fig.update_xaxes(constrain="domain")
    if y_lock is True:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(plot_bgcolor=plot_bgcolor, margin=dict(r=5, l=5, b=5))

    if title:
        fig.update_layout(title=title)

    return fig


decision_tree_card = dbc.Card(
    [
        dbc.CardHeader("Decision Tree Explainer", className="bg-primary text-white"),
        html.Br(style={"margin-bottom": "20px"}),
        dbc.Label("Max Depth"),
        dcc.Dropdown(
            id="max-depth-selector",
            options=[{"label": depth, "value": depth} for depth in list(range(1, 8))],
            value=3,
        ),
        dcc.Graph(id="decision-tree-plot"),
    ]
)

###### Permutation and Feature Importance Plots

clf_path = os.path.join(os.getcwd(), "models", "rf_clf.joblib")
rf_clf = joblib.load(open(clf_path, "rb"))

permutation_importance = np.load(os.path.join(model_path, "permutation_imp.npy"))
perm_sorted_idx = permutation_importance.argsort()

tree_importance_sorted_idx = np.argsort(rf_clf.feature_importances_)
tree_indices = np.arange(0, len(rf_clf.feature_importances_)) + 0.5


def permutation_importance_plot():
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=permutation_importance[perm_sorted_idx],
            y=X_train.columns[perm_sorted_idx],
            orientation="h",
            marker_color="#329932",
        )
    )
    fig.update_layout(
        # title="Permutation Importances",
        yaxis=dict(title="Features"),
        xaxis=dict(title="Permutation Importance mean"),
        height=650,
        # template="simple_white",
    )
    return fig


def feature_importance_plot():
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=rf_clf.feature_importances_[tree_importance_sorted_idx],
            y=tree_indices,
            orientation="h",
            marker_color="#329932",
        )
    )
    fig.update_layout(
        # title="Random Forest Feature Importances",
        yaxis=dict(
            title="Features",
            tickmode="array",
            tickvals=tree_indices,
            ticktext=X_train.columns[tree_importance_sorted_idx],
        ),
        xaxis=dict(title="Feature Importances"),
        height=650,
        # template="simple_white",
    )
    return fig


permutation_imp_card = dbc.Card(
    [
        dbc.CardHeader("Permutation Importance", className="bg-primary text-white"),
        dcc.Graph(id="permutation-plot", figure=permutation_importance_plot()),
    ]
)

feature_imp_card = dbc.Card(
    [
        dbc.CardHeader(
            "Random Forest Feature Importance", className="bg-primary text-white"
        ),
        dcc.Graph(id="feature-importance-plot", figure=feature_importance_plot()),
    ]
)

# Dash app for predicting customer purchasing intention
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        ### first row
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    html.H1(
                        "Online Shoppers Purchasing Intension Prediction",
                        className="text-center text-primary mb-4",
                    ),
                )
            ],
            # style={"margin": "20px"},
        ),
        html.Br(),
        ### second row
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("By Bhola Prasad"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.A(
                                            "Website",
                                            href="https://www.lifewithdata.com/",
                                            target="_blank",
                                        )
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.A(
                                            "LinkedIn",
                                            href="https://www.linkedin.com/in/bhola-prasad-0065834b/",
                                            target="_blank",
                                        )
                                    ]
                                ),
                            ]
                        )
                        # dcc.Markdown(
                        #     "##### Website - [Life With Data](https://www.lifewithdata.com/)",
                        #     className="mb-4",
                        # ),
                    ],
                )
            ],
            justify="start",
        ),
        html.Br(style={"margin-bottom": "50px"}),
        ### Third row
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(
                    [class_output_card, dcc.Graph(id="shap-waterfall-plot")],
                    md=8,
                ),
            ],
            # align="center",
        ),
        ### Fourth Row
        html.Br(style={"margin-bottom": "50px"}),
        dbc.Row([dbc.Col(decision_tree_card)]),
        ### Fifth Row
        html.Br(style={"margin-bottom": "50px"}),
        dbc.Row(
            [dbc.Col(permutation_imp_card, md=5), dbc.Col(feature_imp_card, md=5)],
            justify="between",
        ),  # 'start', 'center', 'end', 'around' and 'between'.
    ],
    fluid=True,
)

# update class output card
@app.callback(
    Output("predicted-class", "children"),
    [
        Input("page-value-input", "value"),
        Input("exit-rate-input", "value"),
        Input("product-related-input", "value"),
        Input("product-related-duration-input", "value"),
        Input("bounce-rate-input", "value"),
        Input("visitor-type-selector", "value"),
        Input("month-selector", "value"),
    ],
)
def predict_purchase_intension(
    PageValues,
    ExitRates,
    ProductRelated,
    ProductRelated_Duration,
    BounceRates,
    VisitorType,
    Month,
):

    # load model

    model_path = os.path.join(os.getcwd(), "models", "rf_rnd_search1.joblib")
    model = joblib.load(open(model_path, "rb"))

    visitor_type_index = np.where(X_train_sel.columns == VisitorType)[0][0]
    month_index = np.where(X_train_sel.columns == Month)[0][0]

    x = np.zeros(len(X_train_sel.columns))
    x[0] = float(PageValues)
    x[1] = float(ExitRates)
    x[2] = float(ProductRelated)
    x[3] = float(ProductRelated_Duration)
    x[4] = float(BounceRates)
    x[visitor_type_index] = 1
    x[month_index] = 1

    return f"Class: {model.predict([x])[0]}"


# visualize decision tree
@app.callback(
    Output("decision-tree-plot", "figure"), [Input("max-depth-selector", "value")]
)
def visualize_tree(max_depth):
    tree_clf = DecisionTreeClassifier(max_depth=max_depth)
    tree_clf.fit(X_train_sel, y_train)
    dot_data = export_graphviz(
        tree_clf,
        out_file=None,
        filled=True,
        rounded=True,
        feature_names=cols_to_use,
        class_names=["0", "1"],
        proportion=True,
        rotate=False,
        precision=2,
    )

    pydot_graph = pydot.graph_from_dot_data(dot_data)[0]
    svg_bytes = pydot_graph.create_svg()
    fig = svg_to_fig(svg_bytes)  # title="Decision Tree Explanation"

    return fig


# plot shap values waterfall chart


@app.callback(
    Output("shap-waterfall-plot", "figure"),
    [
        Input("page-value-input", "value"),
        Input("exit-rate-input", "value"),
        Input("product-related-input", "value"),
        Input("product-related-duration-input", "value"),
        Input("bounce-rate-input", "value"),
        Input("visitor-type-selector", "value"),
        Input("month-selector", "value"),
    ],
)
def shap_waterfall_plot(
    PageValues,
    ExitRates,
    ProductRelated,
    ProductRelated_Duration,
    BounceRates,
    VisitorType,
    Month,
):

    # load model

    model_path = os.path.join(os.getcwd(), "models", "rf_rnd_search1.joblib")
    model = joblib.load(open(model_path, "rb"))

    visitor_type_index = np.where(X_train_sel.columns == VisitorType)[0][0]
    month_index = np.where(X_train_sel.columns == Month)[0][0]

    x = np.zeros(len(X_train_sel.columns))
    x[0] = float(PageValues)
    x[1] = float(ExitRates)
    x[2] = float(ProductRelated)
    x[3] = float(ProductRelated_Duration)
    x[4] = float(BounceRates)
    x[visitor_type_index] = 1
    x[month_index] = 1
    x_series = pd.Series(x, index=cols_to_use)

    # create a tree explainer object
    explainer = shap.TreeExplainer(model)
    # calculate shap values
    shap_values = explainer.shap_values(x_series)

    fill_color = ["#ff0051" if val >= 0 else "#008bfb" for val in shap_values[1]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=shap_values[1],
            y=cols_to_use,
            text=np.round(shap_values[1], 3),
            orientation="h",
            marker_color=fill_color,
        )
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(
            title=f"""P(Buy= {np.round(model.predict_proba([x_series])[0][1],2)})  
            E(Buy= {np.round(explainer.expected_value[1],3)})"""
        ),
        height=600,
        template="simple_white",
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
