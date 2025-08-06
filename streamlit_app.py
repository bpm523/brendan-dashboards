import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from pycoingecko import CoinGeckoAPI
from pybaseball import batting_stats

# Initialize CoinGecko client
gg = CoinGeckoAPI()

# ——— CACHING FUNCTIONS ———
@st.cache_data(ttl=60)
def load_global():
    return gg.get_global()

@st.cache_data(ttl=300)
def load_coins(top_n=50, ids=None):
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': top_n,
        'page': 1,
        'price_change_percentage': '24h,7d'
    }
    if ids:
        params['ids'] = ','.join(ids)
        params['per_page'] = len(ids)
    return gg.get_coins_markets(**params)

@st.cache_data(ttl=300)
def load_categories():
    return gg.get_coins_categories()

@st.cache_data(ttl=300)
def load_trending_coins():
    trending = gg.get_search_trending().get('coins', [])
    ids = [t['item']['id'] for t in trending]
    if not ids:
        return pd.DataFrame()
    data = load_coins(ids=ids)
    df = pd.DataFrame(data)
    df['Rank'] = df['market_cap_rank']
    df['Symbol'] = df['symbol'].str.upper()
    df['Price'] = df['current_price']
    df['24h % Change'] = df['price_change_percentage_24h_in_currency']
    return df[['Rank', 'Symbol', 'name', 'Price', '24h % Change']]

@st.cache_data(ttl=600)
def fetch_quakes(period='day'):
    url = f'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_{period}.geojson'
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        feats = data.get('features', [])
        records = []
        for f in feats:
            props = f['properties']
            coords = f['geometry']['coordinates']
            records.append({
                'Place': props.get('place',''),
                'Magnitude': props.get('mag',0),
                'Time': pd.to_datetime(props.get('time',0), unit='ms'),
                'Depth': coords[2],
                'Longitude': coords[0],
                'Latitude': coords[1]
            })
        return pd.DataFrame(records)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_gapminder():
    return px.data.gapminder()

# ——— HELPERS ———
def human_format(num):
    if num >= 1e12: return f"${num/1e12:,.2f}T"
    if num >= 1e9:  return f"${num/1e9:,.2f}B"
    if num >= 1e6:  return f"${num/1e6:,.2f}M"
    if num >= 1e3:  return f"${num/1e3:,.2f}K"
    return f"${num:,.2f}"

# ——— CRYPTO DASHBOARD ———
def crypto_chart():
    st.header("🌐 Crypto Metrics Dashboard")
    st.markdown("**Live top 50 cryptos: global stats, treemap breakdown, gainers/losers, categories & trending.**")

    # Global Overview
    st.subheader("Global Market Overview")
    gd = load_global()
    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Total Market Cap", human_format(gd['total_market_cap']['usd']))
    c1.metric("24h Volume", human_format(gd['total_volume']['usd']))
    c2.metric("24h Cap Change", f"{gd['market_cap_change_percentage_24h_usd']:.2f}%")
    c3.metric("BTC Dominance", f"{gd['market_cap_percentage']['btc']:.2f}%")

    st.markdown("---")
    # Treemap Breakdown
    st.subheader("Asset Breakdown (Treemap)")
    coins = pd.DataFrame(load_coins())
    coins['Symbol'] = coins['symbol'].str.upper()
    coins['Market Cap'] = coins['market_cap']
    coins['24h Volume'] = coins['total_volume']
    coins['Price'] = coins['current_price']
    coins['24h % Change'] = coins['price_change_percentage_24h_in_currency']
    coins['7d % Change'] = coins['price_change_percentage_7d_in_currency']

    size_m = st.selectbox("Size Metric", ['Market Cap','24h Volume','Price'], index=1)
    color_m = st.selectbox("Color Metric (price % change)", ['24h % Change','7d % Change'], index=0)
    fig = px.treemap(
        coins, path=['Symbol'], values=size_m, color=color_m,
        color_continuous_scale='RdYlGn', hover_data={size_m:':,.0f', color_m:':.2f%'},
        template='plotly_dark'
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), coloraxis_colorbar_tickformat='.2f%')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # Gainers & Losers
    st.subheader("Top 10 Gainers & Losers (24h %)")
    gainers = coins.nlargest(10,'24h % Change')[['symbol','24h % Change']]
    losers  = coins.nsmallest(10,'24h % Change')[['symbol','24h % Change']]
    gainers['24h % Change'] = gainers['24h % Change'].map(lambda x: f"{x:.2f}%")
    losers ['24h % Change'] = losers ['24h % Change'].map(lambda x: f"{x:.2f}%")
    colg, coll = st.columns(2)
    with colg:
        st.markdown("**Gainers**")
        st.table(gainers.rename(columns={'symbol':'Symbol'}).set_index('Symbol'))
    with coll:
        st.markdown("**Losers**")
        st.table(losers.rename(columns={'symbol':'Symbol'}).set_index('Symbol'))

    st.markdown("---")
    # Category Distribution
    st.subheader("Market Cap by Category")
    cats = pd.DataFrame(load_categories())[['name','market_cap']].rename(columns={'name':'Category','market_cap':'Value'})
    cats = cats.nlargest(10,'Value').sort_values('Value',ascending=False)
    cats['Formatted'] = cats['Value'].apply(human_format)
    bar = px.bar(
        cats, x='Value', y='Category', orientation='h', text='Formatted',
        template='plotly_dark', labels={'Value':'Market Cap','Category':'Category'}
    )
    bar.update_traces(textposition='inside', insidetextanchor='start', textfont_size=12)
    bar.update_layout(margin=dict(l=180, r=20, t=10, b=10), yaxis=dict(automargin=True, categoryorder='total descending'))
    st.plotly_chart(bar, use_container_width=True)

    st.markdown("---")
    # Trending Coins
    st.subheader("🚀 Top Trending Coins")
    st.markdown("Sorted by CoinGecko search popularity (last 24h). Shows market cap rank, live USD price & 24h price change.")
    df_trend = load_trending_coins()
    if not df_trend.empty:
        df_trend['Price'] = df_trend['Price'].apply(lambda x: f"${x:,.2f}")
        df_trend['24h % Change'] = df_trend['24h % Change'].apply(lambda x: f"{x:.2f}%")
        st.table(df_trend.set_index('Rank'))
    else:
        st.info("No trending data.")

# ——— EARTHQUAKE DASHBOARD ———
def earthquake_dashboard():
    st.header("🌎 Earthquake Dashboard")
    st.markdown("Recent seismic activity worldwide with magnitude filtering & visualizations.")

    period = st.selectbox("Period", ['hour','day','week','month'], index=1)
    min_mag = st.slider("Minimum Magnitude", 0.0, 8.0, 2.5, 0.1)

    df = fetch_quakes(period)
    if df.empty:
        st.warning("No earthquake data available.")
    else:
        df = df[df['Magnitude'] >= min_mag]
        st.subheader(f"Recent Quakes (Mag ≥ {min_mag}, past {period})")

        fig_map = px.scatter_geo(
            df, lat='Latitude', lon='Longitude', size='Magnitude', color='Depth',
            hover_name='Place', hover_data={'Magnitude':':.1f','Depth':':.1f'},
            projection='natural earth', template='plotly_dark', title='Earthquake Locations'
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.subheader("Magnitude Distribution Over Time")
        df_time = df.set_index('Time').resample('6H').size().reset_index(name='Count')
        fig_ts = px.line(
            df_time, x='Time', y='Count', template='plotly_dark',
            title=f"Number of Quakes every 6 hours (past {period})"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("Top 10 Largest Quakes")
        top10 = df.nlargest(10, 'Magnitude')[['Time','Place','Magnitude','Depth']]
        top10['Time'] = top10['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.table(top10.set_index('Time'))

# ——— GLOBAL DEVELOPMENT DASHBOARD ———
def gapminder_dashboard():
    st.header("🌍 Global Development Dashboard")
    st.markdown("Explore GDP per capita vs life expectancy across countries, over time, and by continent.")
    df_g = load_gapminder()
    min_year, max_year = int(df_g.year.min()), int(df_g.year.max())
    year = st.slider("Year", min_year, max_year, max_year)
    df_year = df_g[df_g.year == year]

    # Scatter: GDP vs Life Expectancy
    fig = px.scatter(
        df_year, x="gdpPercap", y="lifeExp", size="pop", color="continent",
        hover_name="country", log_x=True, size_max=60,
        title=f"Life Expectancy vs GDP per Capita ({year})",
        labels={"gdpPercap":"GDP per Capita (USD)", "lifeExp":"Life Expectancy (yrs)"},
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar: Avg lifeExp by Continent
    st.subheader("Average Life Expectancy by Continent")
    cont = df_year.groupby('continent')['lifeExp'].mean().reset_index()
    fig_bar = px.bar(
        cont, x='lifeExp', y='continent', orientation='h', text='lifeExp',
        labels={'lifeExp':'Avg Life Expectancy','continent':'Continent'},
        template='plotly_dark', title='Avg Life Expectancy'
    )
    fig_bar.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

    # Top 10 by GDP per Capita
    st.subheader("Top 10 Countries by GDP per Capita")
    top10 = df_year.nlargest(10, "gdpPercap").copy()
    top10['gdpPercap'] = top10['gdpPercap'].apply(lambda x: f"${x:,.0f}")
    st.table(top10[['country','gdpPercap']].rename(columns={'country':'Country','gdpPercap':'GDP per Capita'}).set_index('Country'))

# ——— BASEBALL DASHBOARD ———
def baseball_dashboard():
    st.header("⚾ Baseball Analytics (Moneyball Stats)")
    year = st.slider("Season Year", 2015, 2025, 2024)
    stats = ["AVG","OBP","SLG","OPS","WAR","wOBA","ISO","BABIP"]
    stat = st.selectbox("Performance Stat", stats)
    df_b = batting_stats(year)

    # Top 10 by stat
    top10 = df_b[['Name', stat]].nlargest(10, stat).dropna()
    fig = px.bar(
        top10, x=stat, y='Name', orientation='h', text=stat,
        color=stat, color_continuous_scale='Blues', template='plotly_dark',
        labels={stat: stat, 'Name': 'Player'}
    )
    fig.update_traces(textposition='outside', texttemplate='%{text:.2f}')
    st.plotly_chart(fig, use_container_width=True)

    # Distribution & histogram
    st.subheader(f"Distribution of {stat} across players")
    fig2 = px.histogram(
        df_b, x=stat, nbins=30, marginal='box', template='plotly_dark',
        title=f"{stat} Histogram & Boxplot"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Scatter: Age vs Performance
    st.subheader(f"Age vs {stat}")
    df_sc = df_b.dropna(subset=['Age', stat])
    if not df_sc.empty:
        fig3 = px.scatter(
            df_sc, x='Age', y=stat, hover_name='Name', size=stat, color='Age',
            color_continuous_scale='Viridis', template='plotly_dark',
            labels={'Age':'Player Age', stat: stat},
            title=f"Player Age vs {stat} ({year})"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Age data unavailable for scatter.")

# ——— MAIN APP ———
def main():
    st.sidebar.title("📊 Portfolio Dashboards")
    choice = st.sidebar.radio("Choose dashboard:", ["Crypto Metrics","Earthquakes","Global Development","Baseball Analytics"])
    if choice == "Crypto Metrics":
        crypto_chart()
    elif choice == "Earthquakes":
        earthquake_dashboard()
    elif choice == "Global Development":
        gapminder_dashboard()
    else:
        baseball_dashboard()

if __name__ == '__main__':
    main()
