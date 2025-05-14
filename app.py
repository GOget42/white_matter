import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from datetime import datetime

def load_dataset(file_path):
    """Lädt das Dataset und behandelt Fehler"""
    try:
        return xr.open_dataset(file_path)
    except Exception as e:
        st.error(f"Fehler beim Öffnen der Datei: {e}")
        return None

def filter_data_by_scenario_and_time(ds, scenario, start_date, end_date):
    """
    Filtert Daten nach Szenario und Zeitraum.
    Behandelt das Problem mit doppelten Zeitstempeln.
    """
    # Erzeuge eine Liste der Zeitpunkte, die im gewünschten Bereich liegen
    time_mask = (ds.time >= np.datetime64(start_date)) & (ds.time <= np.datetime64(end_date))
    scenario_mask = ds.scenario == scenario

    # Filtere nach Zeit und Szenario gleichzeitig
    filtered_ds = ds.where(time_mask & scenario_mask, drop=True)

    return filtered_ds.snow_depth

def calculate_snow_resource_data(snow_data, params):
    """Berechnet Schneebedarfsdaten für die gegebenen Parameter"""
    monthly_data = []
    month_names = ["Januar", "Februar", "März", "April", "Mai", "Juni",
                   "Juli", "August", "September", "Oktober", "November", "Dezember"]

    for time_point in snow_data.time.values:
        date = pd.to_datetime(time_point)

        # Prüfen, ob der Zeitpunkt in der Skisaison liegt
        if params['season_start_month'] <= params['season_end_month']:
            is_in_season = (date.month >= params['season_start_month']) and (date.month <= params['season_end_month'])
        else:
            # Bei jahresübergreifenden Zeiträumen (z.B. Nov-März)
            is_in_season = (date.month >= params['season_start_month']) or (date.month <= params['season_end_month'])

        if is_in_season:
            # Durchschnittliche Schneehöhe für diesen Zeitpunkt
            avg_snow_depth = float(snow_data.sel(time=time_point).mean(['latitude', 'longitude']).values)

            # Schneebedarf berechnen
            snow_demand_m3 = max(0, (params['min_snow_depth'] - avg_snow_depth) * params['slope_area'])

            # Ressourcenberechnung ohne Zusatzstoff
            water_usage_l = snow_demand_m3 * params['water_per_m3']
            energy_usage_kwh = snow_demand_m3 * params['energy_per_m3']
            total_cost = (water_usage_l * params['water_cost_per_l']) + (energy_usage_kwh * params['energy_cost_per_kwh'])

            # Ressourcenberechnung mit Zusatzstoff
            water_usage_with_additive_l = snow_demand_m3 * (params['water_per_m3'] * (1 - params['additive_efficiency']))
            energy_usage_with_additive_kwh = snow_demand_m3 * (params['energy_per_m3'] * (1 - params['additive_efficiency']))
            total_cost_with_additive = (water_usage_with_additive_l * params['water_cost_per_l']) + \
                                      (energy_usage_with_additive_kwh * params['energy_cost_per_kwh']) + \
                                      (snow_demand_m3 * params['additive_cost_per_m3'])

            # Daten für diese Zeit speichern
            monthly_data.append({
                'Datum': date,
                'Jahr': date.year,
                'Monat': date.month,
                'MonatName': month_names[date.month - 1],
                'DurchschnittlicheSchneehöhe': avg_snow_depth,
                'Schneebedarf_m3': snow_demand_m3,
                'Wasserverbrauch_l': water_usage_l,
                'Wasserverbrauch_mit_Additiv_l': water_usage_with_additive_l,
                'Energieverbrauch_kwh': energy_usage_kwh,
                'Energieverbrauch_mit_Additiv_kwh': energy_usage_with_additive_kwh,
                'Gesamtkosten': total_cost,
                'Gesamtkosten_mit_Additiv': total_cost_with_additive,
                'Kosteneinsparung': total_cost - total_cost_with_additive
            })

    return pd.DataFrame(monthly_data) if monthly_data else pd.DataFrame()

def render_summary_metrics(df, start_date, end_date):
    """Displays the summary metrics"""
    st.subheader(f"Summary for the period {start_date.strftime('%m.%Y')} to {end_date.strftime('%m.%Y')}")
    st.markdown("#### ❄️ Snow")
    st.metric("Total snow requirement", f"{df['Schneebedarf_m3'].sum():,.1f}".replace(",", "'") + " m³")

    col2, col3, col4 = st.columns(3)

    with col2:
        st.markdown("#### 💧 Water")
        ohne = df['Wasserverbrauch_l'].sum() / 1000
        mit = df['Wasserverbrauch_mit_Additiv_l'].sum() / 1000
        einsparung = ohne - mit
        st.metric("Without nucleators", f"{ohne:,.1f}".replace(",", "'") + " m³")
        st.metric("With nucleators", f"{mit:,.1f}".replace(",", "'") + " m³")
        st.metric("Savings", f"{einsparung:,.1f}".replace(",", "'") + " m³",
                  delta=f"{einsparung / ohne * 100:.1f}%" if ohne > 0 else None)

    with col3:
        st.markdown("#### ⚡ Energy")
        ohne = df['Energieverbrauch_kwh'].sum()
        mit = df['Energieverbrauch_mit_Additiv_kwh'].sum()
        einsparung = ohne - mit
        st.metric("Without nucleators", f"{ohne:,.1f}".replace(",", "'") + " kWh")
        st.metric("With nucleators", f"{mit:,.1f}".replace(",", "'") + " kWh")
        st.metric("Savings", f"{einsparung:,.1f}".replace(",", "'") + " kWh",
                  delta=f"{einsparung / ohne * 100:.1f}%" if ohne > 0 else None)

    with col4:
        st.markdown("#### 💰 Costs")
        ohne = df['Gesamtkosten'].sum()
        mit = df['Gesamtkosten_mit_Additiv'].sum()
        einsparung = df['Kosteneinsparung'].sum()
        st.metric("Without nucleators", f"{ohne:,.2f}".replace(",", "'") + " CHF")
        st.metric("With nucleators", f"{mit:,.2f}".replace(",", "'") + " CHF")
        st.metric("Savings", f"{einsparung:,.2f}".replace(",", "'") + " CHF",
                  delta=f"{einsparung / ohne * 100:.1f}%" if ohne > 0 else None)

def plot_monthly_bar_chart(df, y_columns, title, y_axis_title, trace_names, unit_divisor=1, season_start_month=None,
                           season_end_month=None):
    """Universal plotting function for grouped monthly charts that only shows in-season months"""
    # Filtere nur die Daten der Saison
    if season_start_month and season_end_month:
        # Kopiere DataFrame, um das Original nicht zu verändern
        df_season = df.copy()

        # Behalte nur die Einträge, die in der Saison liegen
        if season_start_month <= season_end_month:
            # Normale Saison innerhalb eines Jahres (z.B. Jan-März)
            df_season = df_season[df_season['Monat'].between(season_start_month, season_end_month)]
        else:
            # Jahresübergreifende Saison (z.B. Dez-März)
            df_season = df_season[(df_season['Monat'] >= season_start_month) | (df_season['Monat'] <= season_end_month)]
    else:
        df_season = df

    # Erstelle das Diagramm nur mit den gefilterten Daten
    fig = go.Figure()

    # Erstelle lesbare x-Labels im Format "Monat Jahr"
    df_season['x_label'] = df_season['Datum'].apply(lambda x: x.strftime('%B %Y'))

    # Sortiere die Daten chronologisch
    df_season = df_season.sort_values('Datum')

    for col, name in zip(y_columns, trace_names):
        fig.add_trace(go.Bar(
            x=df_season['x_label'],
            y=df_season[col] / unit_divisor,
            name=name,
            # Speichere das Datum als benutzerdefinierte Daten, um die Sortierung zu erhalten
            customdata=df_season['Datum']
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60),
        # Setze die x-Achse explizit auf die verfügbaren kategorischen Werte
        xaxis=dict(
            type='category',
            categoryorder='array',
            categoryarray=df_season['x_label'].tolist()
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def render_all_charts(df, season_start_month, season_end_month):
    tabs = st.tabs(["❄️ Snow Requirement", "💰 Costs", "💧 Water", "⚡ Energy"])

    with tabs[0]:
        plot_monthly_bar_chart(
            df,
            y_columns=["Schneebedarf_m3"],
            title="Monthly Snow Requirement",
            y_axis_title="Snow Requirement (m³)",
            trace_names=["Standard"],
            season_start_month=season_start_month,
            season_end_month=season_end_month
        )

    with tabs[1]:
        plot_monthly_bar_chart(
            df,
            y_columns=["Gesamtkosten", "Gesamtkosten_mit_Additiv"],
            title="Monthly Costs",
            y_axis_title="Costs (CHF)",
            trace_names=["Without nucleators", "With nucleators"],
            season_start_month=season_start_month,
            season_end_month=season_end_month
        )

    with tabs[2]:
        plot_monthly_bar_chart(
            df,
            y_columns=["Wasserverbrauch_l", "Wasserverbrauch_mit_Additiv_l"],
            title="Monthly Water Consumption",
            y_axis_title="Water Consumption (m³)",
            trace_names=["Without nucleators", "With nucleators"],
            unit_divisor=1000,
            season_start_month=season_start_month,
            season_end_month=season_end_month
        )

    with tabs[3]:
        plot_monthly_bar_chart(
            df,
            y_columns=["Energieverbrauch_kwh", "Energieverbrauch_mit_Additiv_kwh"],
            title="⚡ Monthly Energy Consumption",
            y_axis_title="Energy Consumption (kWh)",
            trace_names=["Without nucleators", "With nucleators"],
            season_start_month=season_start_month,
            season_end_month=season_end_month
        )

def display_detailed_analysis(df):
    """Displays the detailed analysis table"""
    st.subheader("Detailed Analysis")

    detailed_df = df[['Datum', 'DurchschnittlicheSchneehöhe', 'Schneebedarf_m3',
                      'Gesamtkosten', 'Gesamtkosten_mit_Additiv', 'Kosteneinsparung']]

    detailed_df = detailed_df.rename(columns={
        'Datum': 'Date',
        'DurchschnittlicheSchneehöhe': 'Snow Depth (m)',
        'Schneebedarf_m3': 'Snow Requirement (m³)',
        'Gesamtkosten': 'Costs (CHF)',
        'Gesamtkosten_mit_Additiv': 'Costs with Nucleators (CHF)',
        'Kosteneinsparung': 'Savings (CHF)'
    })

    # Table display with formatting
    st.dataframe(detailed_df.style.format({
        'Snow Depth (m)': '{:.2f}',
        'Snow Requirement (m³)': '{:.1f}',
        'Snow Requirement with Nucleators (m³)': '{:.1f}',
        'Costs (CHF)': '{:.2f}',
        'Costs with Nucleators (CHF)': '{:.2f}',
        'Savings (CHF)': '{:.2f}'
    }))

    # CSV download button
    csv = detailed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Data as CSV",
        csv,
        "snow_requirement_analysis.csv",
        "text/csv",
        key='download-csv'
    )

def main():
    # Page configuration for a clean layout
    st.set_page_config(page_title="WAG Nucleator Analysis", layout="wide")

    # Page title and description
    st.title("WAG Nucleator Analysis")
    st.write("Analyze snow requirements and compare costs with and without nucleators.")

    # Automatic loading of NetCDF file
    nc_file_path = "snow_depth_prediction.nc"

    # Check if file exists
    if not os.path.exists(nc_file_path):
        st.error(f"File '{nc_file_path}' not found. Please make sure the file is in the same directory as the app.")
        return

    # Load data
    with st.spinner("Loading data..."):
        ds = load_dataset(nc_file_path)

    if ds is None:
        return

    # Insert logo on sidebar
    st.sidebar.image('https://i.imgur.com/0asnwsn.png', use_container_width=True)
    # Sidebar for user inputs
    st.sidebar.header("Settings")

    # Mapping of scenarios to readable labels
    scenario_labels = {
        "ssp126": "Sustainable Scenario (SSP1-2.6)",
        "ssp245": "Medium Scenario (SSP2-4.5)",
        "ssp370": "High Scenario (SSP3-7.0)",
        "ssp585": "Extreme Scenario (SSP5-8.5)"
    }

    # Get available scenarios and label them
    available_scenarios = [
        scenario_labels.get(scenario, scenario) for scenario in np.unique(ds.scenario.values)
    ] if 'scenario' in ds.coords else ["Standard Scenario"]

    # Scenario selection
    st.sidebar.subheader("🌤️ Scenario")
    chosen_scenario_label = st.sidebar.selectbox("Select climate scenario", available_scenarios)
    chosen_scenario = list(scenario_labels.keys())[list(scenario_labels.values()).index(
        chosen_scenario_label)] if chosen_scenario_label in scenario_labels.values() else chosen_scenario_label

    # Basic inputs with icons
    st.sidebar.subheader("🏔️ Ski Resort")
    min_snow_depth = st.sidebar.number_input("Minimum snow depth for skiing (m)", min_value=0.1, value=0.5, step=0.1)
    slope_area = st.sidebar.number_input("Slope area (m²)", min_value=1000, value=1000000, step=10000)

    # Available winter months for season definition
    winter_months = ["December", "January", "February", "March"]

    # Season data - limited to available data months
    st.sidebar.subheader("📅 Ski Season")
    season_start = st.sidebar.selectbox("Season start", winter_months, index=0)  # Default to December
    season_end = st.sidebar.selectbox("Season end", winter_months, index=3)  # Default to March

    # Convert month names to numbers
    month_names = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    season_start_month = month_names.index(season_start) + 1 if season_start != "December" else 12
    season_end_month = month_names.index(season_end) + 1

    # Forecast period with improved inputs
    st.sidebar.subheader("📊 Analysis Period")

    # Extract available time periods from data
    time_min = pd.to_datetime(ds.time.min().values)
    time_max = pd.to_datetime(ds.time.max().values)

    # Years and months for start and end selection
    current_year = datetime.now().year

    # Available months for analysis period selection
    available_months = ["December", "January", "February", "March"]
    default_month_index = 0  # December as default

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_year = st.number_input("Start year",
                                     min_value=int(time_min.year),
                                     max_value=int(time_max.year),
                                     value=current_year)
    with col2:
        start_month = st.selectbox("Start month", available_months, index=default_month_index)
        start_month = 12 if start_month == "December" else month_names.index(start_month) + 1

    col1, col2 = st.sidebar.columns(2)
    with col1:
        end_year = st.number_input("End year",
                                   min_value=int(time_min.year),
                                   max_value=int(time_max.year),
                                   value=min(current_year + 5, int(time_max.year)))
    with col2:
        end_month = st.selectbox("End month", available_months, index=3)  # Default to March
        end_month = 12 if end_month == "December" else month_names.index(end_month) + 1

    # Additive inputs
    st.sidebar.subheader("🧪 Nucleators")
    additive_efficiency = st.sidebar.slider("Efficiency (% of resource savings)",
                                            min_value=0,
                                            max_value=90,
                                            value=30,
                                            step=1) / 100

    # Cost and resource parameters
    st.sidebar.subheader("💰 Costs & Resources")

    with st.sidebar.expander("Adjust cost parameters"):
        additive_cost_per_m3 = st.number_input("Nucleator cost per m³ of snow(CHF)", min_value=0.001, value=0.050,
                                               step=0.001, format="%.3f")
        water_cost_per_l = st.number_input("Water cost per liter (CHF)", min_value=0.0001, value=0.002, step=0.0005,
                                           format="%.4f")
        energy_cost_per_kwh = st.number_input("Energy cost per kWh (CHF)", min_value=0.01, value=0.25, step=0.01)

    with st.sidebar.expander("Adjust resource parameters"):
        water_per_m3 = st.number_input("Water usage per m³ of snow (l)", min_value=50, value=200, step=10)
        energy_per_m3 = st.number_input("Energy usage per m³ of snow (kWh)", min_value=1.0, value=5.0,
                                        step=0.5)

    # Collect parameters
    params = {
        'min_snow_depth': min_snow_depth,
        'slope_area': slope_area,
        'season_start_month': season_start_month,
        'season_end_month': season_end_month,
        'additive_efficiency': additive_efficiency,
        'water_per_m3': water_per_m3,
        'energy_per_m3': energy_per_m3,
        'water_cost_per_l': water_cost_per_l,
        'energy_cost_per_kwh': energy_cost_per_kwh,
        'additive_cost_per_m3': additive_cost_per_m3
    }

    # Main area: Data processing and display
    # Create date range
    start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
    if start_month == 12 and end_month != 12:
        # Handle December to other month (cross year)
        if start_year == end_year:
            end_year += 1  # Must be next year if starting in December

    if end_month == 12:
        end_date = pd.Timestamp(year=end_year, month=end_month, day=31)
    else:
        end_date = pd.Timestamp(year=end_year, month=end_month, day=1) + pd.Timedelta(days=31)
        end_date = end_date.replace(day=1) - pd.Timedelta(days=1)  # Last day of month

    # Filter data by time period and scenario
    with st.spinner("Processing data..."):
        try:
            # Use new filter function that can handle non-unique timestamps
            snow_data = filter_data_by_scenario_and_time(ds, chosen_scenario, start_date, end_date)
            df = calculate_snow_resource_data(snow_data, params)
        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.info("Try selecting different time periods within December to March only.")
            return

    # Display data if available
    if not df.empty:
        # Summary
        render_summary_metrics(df, start_date, end_date)

        # Charts
        render_all_charts(df, params['season_start_month'], params['season_end_month'])

        # Divider
        st.divider()

        # Detailed table
        display_detailed_analysis(df)
    else:
        st.warning(
            "No data available for the selected parameters. Please ensure your selection includes months from December to March only.")
        st.info(
            "Tip: Check that your selected time range contains winter months (December to March) and falls within the available data period.")

    with st.expander("📘 Click here to learn more about the underlying calculations"):
        st.markdown("## 🔢 Calculation Basics - Example")
        st.markdown(
            'In the following, a calculation example is carried out using assumed parameters to explain the calculations behind the model.')
        st.markdown("### 📌 Assumptions")
        st.markdown("""
    - Climate scenario: RCP 2.6  
    - Analysis period: 05.2025 to 05.2030  
    - Slope area: 1'000'000 m²  
    - Minimum snow depth: 1 m  
    - Efficiency increase through nucleators: 30 %  
    - Cost of nucleators: 0.05 CHF per m³ of snow  
    - Water costs: 0.002 CHF per liter  
    - Energy costs: 0.25 CHF per kWh  
    - Water consumption: 200 liters per m³ of snow  
    - Energy consumption: 5 kWh per m³ of snow  
    """)

        st.markdown("### ❄️ Total Snow Requirement")
        st.markdown(
            "The snow requirement is calculated by subtracting the predicted snowfall according to the selected scenario from the entered minimum snow depth. This difference in meters is then multiplied by the entered slope area.")
        st.latex(r"V = (h_{\text{min}} - h_{\text{nat}}) \times A = 507'156.5 \, \text{m}^3")

        st.markdown("### 💧 Water Consumption")
        st.markdown("**Without nucleators:**")
        st.latex(r"W_{\text{without}} = V \times 200\,\text{Liter} = 507'156.5 \times 0.2 = 101'431.3\,\text{m}^3")

        st.markdown("**With nucleators (30 % savings):**")
        st.latex(r"W_{\text{with}} = V \times (1 - 0.30) \times 0.2 = 71'001.9\,\text{m}^3")

        st.markdown("### ⚡ Energy Consumption")
        st.markdown("**Without nucleators:**")
        st.latex(r"E_{\text{without}} = V \times 5 = 507'156.5 \times 5 = 2'535'782.7\,\text{kWh}")

        st.markdown("**With nucleators (30 % savings):**")
        st.latex(r"E_{\text{with}} = V \times (1 - 0.30) \times 5 = 1'775'047.9\,\text{kWh}")

        st.markdown("### 💰 Costs")
        st.markdown("**Without nucleators:**")
        st.latex(r"""
    K_{\text{without}} = 
    W_{\text{without}} \times 0.002 +
    E_{\text{without}} \times 0.25 =
    101'431.3 \times 0.002 +
    2'535'782.7 \times 0.25 =
    836'808.28 \, \text{CHF}
    """)

        st.markdown("**With nucleators:**")
        st.latex(r"""
    K_{\text{with}} =
    W_{\text{with}} \times 0.002 +
    E_{\text{with}} \times 0.25 +
    V \times (1 - 0.30) \times 0.05 =
    611'123.63 \, \text{CHF}
    """)

        st.markdown("**Savings:**")
        st.latex(r"K_{\text{Savings}} = K_{\text{without}} - K_{\text{with}} = 225'684.66 \, \text{CHF}")
    # Footer
    st.markdown("---")
    st.markdown(
        "<small>© 2025 WhiteMatter Insights | Developed for the Weisse Arena Group</small>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()