import dash
import flask
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.cluster import KMeans

# Load the LLCP2015 dataset into a DataFrame

df = pd.read_csv(r"C:\Users\Eya Kaabachi\Desktop\new_datapfe.csv")

# constants
feature_mapping = {
    'cancer': ['SEX', 'EXERANY2', '_RFBING5', '_RFDRHV5', '_FRTLT1', '_VEGLT1',
               'BPMEDS', 'BLOODCHO', 'ADDEPEV2', 'TOLDHI2', 'CVDINFR4', 'CVDCRHD4',
               'CVDSTRK3', '_MICHD', 'DIABETE3', 'SMOKE100', 'SMOKDAY2', 'USENOW3',
               '_SMOKER3', '_RFSMOK3', 'CHCSCNCR'],
    'diabete': ['GENHLTH', 'EXERANY2', '_RFBING5', '_RFDRHV5', 'DRNKANY5', '_VEGLT1',
                'BPHIGH4', 'BPMEDS', 'BLOODCHO', 'CHOLCHK', 'TOLDHI2', 'CVDINFR4',
                'CVDCRHD4', 'CVDSTRK3', '_MICHD', 'CHCSCNCR', 'SMOKDAY2', 'USENOW3',
                '_SMOKER3', '_RFSMOK3', 'SMOKE100', 'DIABETE3'],
    'heart': ['SEX', 'EXERANY2', '_RFDRHV5', '_FRTLT1', '_VEGLT1', 'BPHIGH4',
              'BPMEDS', 'BLOODCHO', 'ADDEPEV2', 'TOLDHI2', 'CVDINFR4', 'CVDSTRK3',
              '_MICHD', 'CHCSCNCR', 'DIABETE3', 'SMOKE100', 'SMOKDAY2', 'USENOW3',
              '_SMOKER3', '_RFSMOK3', '_RFBING5', 'CVDCRHD4'],
    'mental': ['SEX', 'EXERANY2', '_RFDRHV5', 'MENTHLTH', '_FRTLT1', '_VEGLT1',
               'BPMEDS', 'BLOODCHO', 'TOLDHI2', 'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3',
               '_MICHD', 'CHCSCNCR', 'DIABETE3', 'SMOKE100', 'SMOKDAY2', 'USENOW3',
               '_SMOKER3', '_RFSMOK3', '_RFBING5', 'ADDEPEV2']
}

num_clusters = 3


# Function to generate the Dash app based on selected main feature
def create_cluster_column(main_feature: str) -> pd.DataFrame:
    # Map main features to the corresponding cluster features

    # Select the features for clustering based on the main feature
    features_for_clustering = feature_mapping[main_feature]

    # Normalize the data for clustering
    df_normalized = df[features_for_clustering].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Use k-means clustering to create clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_normalized)
    return df


def get_context_layout(main_feature: str) -> html.Div:
    context_dataframe = create_cluster_column(main_feature=main_feature)
    return html.Div([
        html.Div([
            html.Div([
                html.H3(f"Cluster {cluster}", style={'textAlign': 'center', 'color': '#4e73df'}),
                dcc.Graph(
                    id=f'gender-pie-chart-cluster-{cluster}',
                    figure={
                        'data': [
                            go.Pie(
                                labels=['Male', 'Female'],
                                values=context_dataframe[context_dataframe['Cluster'] == cluster]['SEX'].value_counts(),
                                hole=0.5,
                                marker={'colors': ['#1f77b4', '#1cc88a']},
                                hoverinfo='label+percent'
                            )
                        ],
                        'layout': go.Layout(
                            title='Gender Distribution',
                            legend=dict(orientation='h'),
                            height=350
                        )
                    },
                    style={'margin': '0 auto'}
                ),

                dcc.Graph(
                    id=f'bar-chart-cluster-{cluster}',
                    figure={
                        'data': [
                            go.Bar(
                                x=['Smoking', 'Drinking', 'Exercising', main_feature.capitalize()],
                                y=[
                                    context_dataframe[context_dataframe['Cluster'] == cluster]['SMOKE100'].sum(),
                                    context_dataframe[context_dataframe['Cluster'] == cluster]['_RFBING5'].sum(),
                                    context_dataframe[context_dataframe['Cluster'] == cluster]['EXERANY2'].sum(),
                                    context_dataframe[context_dataframe['Cluster'] == cluster][
                                        feature_mapping[main_feature][-1]].sum()
                                ],
                                marker={'color': ['#36b9cc', '#1cc88a', '#f6c23e', '#4e73df']},
                                hoverinfo='y',
                            )
                        ],
                        'layout': go.Layout(
                            title='Health Status',
                            xaxis={'title': 'Category'},
                            yaxis={'title': 'Number of People'},
                            height=350
                        )
                    },
                    style={'margin': '0 auto'}
                )
            ], style={'width': '40%', 'display': 'inline-block', 'padding': '20px', 'textAlign': 'center'})
            for cluster in range(num_clusters)
        ])
    ])


server = flask.Flask(__name__)  # Creating the Flask server

app = dash.Dash(__name__, server=server)  # Connecting the Dash app to the Flask server
app.css.append_css({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
})

"""stay"""
# Defining the layout of the app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Location component to track the URL
    html.Div(id='content-container')  # Container to display the selected content
])


# Callback to update the content based on the URL endpoint
@app.callback(Output('content-container', 'children'), [Input('url', 'pathname')])
def update_content(pathname):
    feature = pathname[1:]
    if feature in ["cancer", "diabete", "heart", "mental"]:
        return get_context_layout(main_feature=pathname[1:])
    else:
        # TODO Handle wrong pathname
        return html.Div([])


# Run the app
if __name__ == '__main__':
    app.run_server(port=1111)
