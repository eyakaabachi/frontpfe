import dash
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from sklearn.cluster import KMeans

# Load the LLCP2015 dataset into a DataFrame

df = pd.read_csv(r"C:\Users\Eya Kaabachi\Desktop\new_datapfe.csv")


# Function to generate the Dash app based on selected main feature
def generate_dash_app(main_feature):
    # Map main features to the corresponding cluster features
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

    if main_feature not in feature_mapping:
        raise ValueError("Invalid main feature.")

    # Select the features for clustering based on the main feature
    features_for_clustering = feature_mapping[main_feature]

    # Normalize the data for clustering
    df_normalized = df[features_for_clustering].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Use k-means clustering to create clusters
    num_clusters = 3  # You can set the desired number of clusters here
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_normalized)

    # Create the Dash app
    app = dash.Dash(__name__)

    # Define CSS styles
    app.css.append_css({
        'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
    })

    # Define the layout of the app
    app.layout = html.Div([
        html.Div([
            html.Div([
                html.H3(f"Cluster {cluster}", style={'textAlign': 'center', 'color': '#4e73df'}),
                dcc.Graph(
                    id=f'gender-pie-chart-cluster-{cluster}',
                    figure={
                        'data': [
                            go.Pie(
                                labels=['Male', 'Female'],
                                values=df[df['Cluster'] == cluster]['SEX'].value_counts(),
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
                                    df[df['Cluster'] == cluster]['SMOKE100'].sum(),
                                    df[df['Cluster'] == cluster]['_RFBING5'].sum(),
                                    df[df['Cluster'] == cluster]['EXERANY2'].sum(),
                                    df[df['Cluster'] == cluster][feature_mapping[main_feature][-1]].sum()
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

    return app


# Run the Dash application
if __name__ == '__main__':
    # Replace 'cancer', 'diabetes', 'heart', or 'mental' with the desired main feature

    generate_dash_app(main_feature='diabete').run_server( port=9999)
