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
            # Bei saisonübergreifenden Zeiträumen (z.B. Nov-März)
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
    """Zeigt die Zusammenfassungsmetriken an"""
    with st.expander("📘 Klicken Sie hier, um mehr über die dahinterstehenden Berechnungen zu erfahren"):
    

        st.markdown("## 🔢 Berechnungsgrundlagen")
        st.markdown('Nachfolgend wird mit Hilfe angenommener Parameter ein Berechnungsbeispiel durchgeführt, um die hinter dem Modell stehenden Berechnungen zu erklären.')
        st.markdown("### 📌 Annahmen")
        st.markdown("""
    - Klimaszenario: RCP 2.6  
    - Analysezeitraum: 05.2025 bis 05.2030  
    - Pistenfläche: 1'000'000 m²  
    - Mindestschneehöhe: 1 m  
    - Effizienzsteigerung durch Keimbildner: 30 %  
    - Kosten Keimbildner: 0.05 CHF pro m³ Schnee  
    - Wasserkosten: 0.002 CHF pro Liter  
    - Energiekosten: 0.25 CHF pro kWh  
    - Wasserverbrauch: 200 Liter pro m³ Schnee  
    - Energieverbrauch: 5 kWh pro m³ Schnee  
    """)

        st.markdown("### ❄️ Gesamter Schneebedarf")
        st.markdown("Der Schneebedarf wird berechnet, indem die eingegebenen Mindestschneehöhe vom prognostizierten Schneefall gemäss ausgewähltem Szenario subtrahiert wird. Diese Differenz in Meter wird im Anschluss mit der eingegebenen Pistenfläche multipliziert.")
        st.latex(r"V = (h_{\text{min}} - h_{\text{nat}}) \times A = 507'156.5 \, \text{m}^3")

        st.markdown("### 💧 Wasserverbrauch")
        st.markdown("**Ohne Keimbildner:**")
        st.latex(r"W_{\text{ohne}} = V \times 200\,\text{Liter} = 507'156.5 \times 0.2 = 101'431.3\,\text{m}^3")

        st.markdown("**Mit Keimbildner (30 % Ersparnis):**")
        st.latex(r"W_{\text{mit}} = V \times (1 - 0.30) \times 0.2 = 71'001.9\,\text{m}^3")

        st.markdown("### ⚡ Energieverbrauch")
        st.markdown("**Ohne Keimbildner:**")
        st.latex(r"E_{\text{ohne}} = V \times 5 = 507'156.5 \times 5 = 2'535'782.7\,\text{kWh}")

        st.markdown("**Mit Keimbildner (30 % Ersparnis):**")
        st.latex(r"E_{\text{mit}} = V \times (1 - 0.30) \times 5 = 1'775'047.9\,\text{kWh}")

        st.markdown("### 💰 Kosten")
        st.markdown("**Ohne Keimbildner:**")
        st.latex(r"""
    K_{\text{ohne}} = 
    W_{\text{ohne}} \times 0.002 +
    E_{\text{ohne}} \times 0.25 =
    101'431.3 \times 0.002 +
    2'535'782.7 \times 0.25 =
    836'808.28 \, \text{CHF}
    """)

        st.markdown("**Mit Keimbildner:**")
        st.latex(r"""
    K_{\text{mit}} =
    W_{\text{mit}} \times 0.002 +
    E_{\text{mit}} \times 0.25 +
    V \times (1 - 0.30) \times 0.05 =
    611'123.63 \, \text{CHF}
    """)

        st.markdown("**Ersparnis:**")
        st.latex(r"K_{\text{Ersparnis}} = K_{\text{ohne}} - K_{\text{mit}} = 225'684.66 \, \text{CHF}")


    st.subheader(f"Zusammenfassung für den Zeitraum {start_date.strftime('%m.%Y')} bis {end_date.strftime('%m.%Y')}")
    st.markdown("#### ❄️ Schnee")
    st.metric("Gesamter Schneebedarf", f"{df['Schneebedarf_m3'].sum():,.1f}".replace(",", "'") + " m³")

    col2, col3, col4 = st.columns(3)


    with col2:
        st.markdown("#### 💧 Wasser")
        ohne = df['Wasserverbrauch_l'].sum() / 1000
        mit = df['Wasserverbrauch_mit_Additiv_l'].sum() / 1000
        einsparung = ohne - mit
        st.metric("Ohne Keimbildner", f"{ohne:,.1f}".replace(",", "'") + " m³")
        st.metric("Mit Keimbildner", f"{mit:,.1f}".replace(",", "'") + " m³")
        st.metric("Ersparnis", f"{einsparung:,.1f}".replace(",", "'") + " m³",
                  delta=f"{einsparung / ohne * 100:.1f}%" if ohne > 0 else None)

    with col3:
        st.markdown("#### ⚡ Energie")
        ohne = df['Energieverbrauch_kwh'].sum()
        mit = df['Energieverbrauch_mit_Additiv_kwh'].sum()
        einsparung = ohne - mit
        st.metric("Ohne Keimbildner", f"{ohne:,.1f}".replace(",", "'") + " kWh")
        st.metric("Mit Keimbildner", f"{mit:,.1f}".replace(",", "'") + " kWh")
        st.metric("Ersparnis", f"{einsparung:,.1f}".replace(",", "'") + " kWh",
                  delta=f"{einsparung / ohne * 100:.1f}%" if ohne > 0 else None)

    with col4:
        st.markdown("#### 💰 Kosten")
        ohne = df['Gesamtkosten'].sum()
        mit = df['Gesamtkosten_mit_Additiv'].sum()
        einsparung = df['Kosteneinsparung'].sum()
        st.metric("Ohne Keimbildner", f"{ohne:,.2f}".replace(",", "'") + " CHF")
        st.metric("Mit Keimbildner", f"{mit:,.2f}".replace(",", "'") + " CHF")
        st.metric("Ersparnis", f"{einsparung:,.2f}".replace(",", "'") + " CHF",
                  delta=f"{einsparung / ohne * 100:.1f}%" if ohne > 0 else None)

def plot_monthly_bar_chart(df, y_columns, title, y_axis_title, trace_names, unit_divisor=1):
    """Universelle Plot-Funktion für gruppierte Monatsdiagramme"""
    fig = go.Figure()
    for col, name in zip(y_columns, trace_names):
        fig.add_trace(go.Bar(
            x=df['Datum'],
            y=df[col] / unit_divisor,
            name=name
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Datum",
        yaxis_title=y_axis_title,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60)
    )
    st.plotly_chart(fig, use_container_width=True)

def render_all_charts(df):
    tabs = st.tabs(["❄️ Schneebedarf", "💰 Kosten", "💧 Wasser", "⚡ Energie"])

    with tabs[0]:
        plot_monthly_bar_chart(
            df,
            y_columns=["Schneebedarf_m3"],
            title="Monatlicher Schneebedarf",
            y_axis_title="Schneebedarf (m³)",
            trace_names=["Standard"]
        )

    with tabs[1]:
        plot_monthly_bar_chart(
            df,
            y_columns=["Gesamtkosten", "Gesamtkosten_mit_Additiv"],
            title="Monatliche Kosten",
            y_axis_title="Kosten (CHF)",
            trace_names=["Ohne Keimbildner", "Mit Keimbildner"]
        )

    with tabs[2]:
        plot_monthly_bar_chart(
            df,
            y_columns=["Wasserverbrauch_l", "Wasserverbrauch_mit_Additiv_l"],
            title="Monatlicher Wasserverbrauch",
            y_axis_title="Wasserverbrauch (m³)",
            trace_names=["Ohne Keimbildner", "Mit Keimbildner"],
            unit_divisor=1000
        )

    with tabs[3]:
        plot_monthly_bar_chart(
            df,
            y_columns=["Energieverbrauch_kwh", "Energieverbrauch_mit_Additiv_kwh"],
            title="⚡ Monatlicher Energieverbrauch",
            y_axis_title="Energieverbrauch (kWh)",
            trace_names=["Ohne Keimbildner", "Mit Keimbildner"]
        )

def display_detailed_analysis(df):
    """Zeigt die detaillierte Analysetabelle an"""
    st.subheader("Detailanalyse")

    detailed_df = df[['Datum', 'DurchschnittlicheSchneehöhe', 'Schneebedarf_m3',
                      'Gesamtkosten', 'Gesamtkosten_mit_Additiv', 'Kosteneinsparung']]

    detailed_df = detailed_df.rename(columns={
        'Datum': 'Datum',
        'DurchschnittlicheSchneehöhe': 'Schneehöhe (m)',
        'Schneebedarf_m3': 'Schneebedarf (m³)',
        'Gesamtkosten': 'Kosten (CHF)',
        'Gesamtkosten_mit_Additiv': 'Kosten mit Keimbildner (CHF)',
        'Kosteneinsparung': 'Einsparung (CHF)'
    })

    # Tabelle anzeigen mit Formatierung
    st.dataframe(detailed_df.style.format({
        'Schneehöhe (m)': '{:.2f}',
        'Schneebedarf (m³)': '{:.1f}',
        'Schneebedarf mit Keimbildner (m³)': '{:.1f}',
        'Kosten (CHF)': '{:.2f}',
        'Kosten mit Keimbildner (CHF)': '{:.2f}',
        'Einsparung (CHF)': '{:.2f}'
    }))

    # CSV-Download-Button
    csv = detailed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Daten als CSV herunterladen",
        csv,
        "schneebedarf_analyse.csv",
        "text/csv",
        key='download-csv'
    )

def main():
    # Seitenkonfiguration für ein sauberes Layout
    st.set_page_config(page_title="WAG Keimbildner Analyse", layout="wide")

    # Seitentitel und Beschreibung
    st.title("WAG Keimbildner Analyse")
    st.write("Analysieren Sie den Schneebedarf und vergleichen Sie Kosten mit und ohne Keimbildner.")

    # Automatisches Laden der NetCDF-Datei
    nc_file_path = "snow_depth_prediction.nc"

    # Prüfen, ob die Datei existiert
    if not os.path.exists(nc_file_path):
        st.error(f"Die Datei '{nc_file_path}' wurde nicht gefunden. Bitte stellen Sie sicher, dass die Datei im gleichen Verzeichnis wie die App liegt.")
        return

    # Daten laden
    with st.spinner("Lade Daten..."):
        ds = load_dataset(nc_file_path)

    if ds is None:
        return

    # Logo auf Sidebar einfügen
    st.sidebar.image('https://i.imgur.com/pnZ1HBn.png', use_column_width=True)
    # Sidebar für Benutzereingaben
    st.sidebar.header("Einstellungen")

    # Mapping der Szenarien auf lesbare Labels
    scenario_labels = {
        "ssp126": "Nachhaltiges Szenario (SSP1-2.6)",
        "ssp245": "Mittleres Szenario (SSP2-4.5)",
        "ssp370": "Hohes Szenario (SSP3-7.0)",
        "ssp585": "Extremes Szenario (SSP5-8.5)"
    }

    # Verfügbare Szenarien abrufen und mit Labels versehen
    available_scenarios = [
        scenario_labels.get(scenario, scenario) for scenario in np.unique(ds.scenario.values)
    ] if 'scenario' in ds.coords else ["Standardszenario"]

    # Szenario-Auswahl
    st.sidebar.subheader("🌤️ Szenario")
    chosen_scenario_label = st.sidebar.selectbox("Klimaszenario auswählen", available_scenarios)
    chosen_scenario = list(scenario_labels.keys())[list(scenario_labels.values()).index(
        chosen_scenario_label)] if chosen_scenario_label in scenario_labels.values() else chosen_scenario_label

    # Basiseingaben mit Icons
    st.sidebar.subheader("🏔️ Skigebiet")
    min_snow_depth = st.sidebar.number_input("Mindestschneehöhe für Skifahren (m)", min_value=0.1, value=0.5, step=0.1)
    slope_area = st.sidebar.number_input("Pistenfläche (m²)", min_value=1000, value=1000000, step=10000)

    # Saisondaten
    st.sidebar.subheader("📅 Skisaison")
    season_start = st.sidebar.selectbox("Saisonbeginn", ["Januar", "Februar", "März", "April", "Mai", "Juni",
                                                         "Juli", "August", "September", "Oktober", "November",
                                                         "Dezember"], 10)
    season_end = st.sidebar.selectbox("Saisonende", ["Januar", "Februar", "März", "April", "Mai", "Juni",
                                                     "Juli", "August", "September", "Oktober", "November", "Dezember"],
                                      3)

    # Monatsnamen in Zahlen umwandeln
    month_names = ["Januar", "Februar", "März", "April", "Mai", "Juni",
                   "Juli", "August", "September", "Oktober", "November", "Dezember"]
    season_start_month = month_names.index(season_start) + 1
    season_end_month = month_names.index(season_end) + 1

    # Prognosezeitraum mit verbesserten Eingaben
    st.sidebar.subheader("📊 Analysezeitraum")

    # Verfügbare Zeiträume aus den Daten extrahieren
    time_min = pd.to_datetime(ds.time.min().values)
    time_max = pd.to_datetime(ds.time.max().values)

    # Jahre und Monate für Start- und Endauswahl
    current_year = datetime.now().year
    current_month = datetime.now().month

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_year = st.number_input("Startjahr",
                                     min_value=int(time_min.year),
                                     max_value=int(time_max.year),
                                     value=current_year)
    with col2:
        start_month = st.selectbox("Startmonat", month_names, index=current_month - 1)
        start_month = month_names.index(start_month) + 1

    col1, col2 = st.sidebar.columns(2)
    with col1:
        end_year = st.number_input("Endjahr",
                                   min_value=int(time_min.year),
                                   max_value=int(time_max.year),
                                   value=min(current_year + 5, int(time_max.year)))
    with col2:
        end_month = st.selectbox("Endmonat", month_names, index=current_month - 1)
        end_month = month_names.index(end_month) + 1

    # Zusatzstoffeingaben
    st.sidebar.subheader("🧪 Keimbildner")
    additive_efficiency = st.sidebar.slider("Effizenz (in % der Ressourcenersparnis)",
                                            min_value=0,
                                            max_value=90,
                                            value=30,
                                            step=1) / 100

    # Kosten- und Ressourcenparameter
    st.sidebar.subheader("💰 Kosten & Ressourcen")

    with st.sidebar.expander("Kostenparameter anpassen"):
        additive_cost_per_m3 = st.number_input("Zusatzstoffkosten pro m³ (CHF)", min_value=0.001, value=0.050,
                                               step=0.001, format="%.3f")
        water_cost_per_l = st.number_input("Wasserkosten pro Liter (CHF)", min_value=0.0001, value=0.002, step=0.0005,
                                           format="%.4f")
        energy_cost_per_kwh = st.number_input("Energiekosten pro kWh (CHF)", min_value=0.01, value=0.25, step=0.01)

    with st.sidebar.expander("Ressourcenparameter anpassen"):
        water_per_m3 = st.number_input("Wasserverbrauch pro m³ Kunstschnee (l)", min_value=50, value=200, step=10)
        energy_per_m3 = st.number_input("Energieverbrauch pro m³ Kunstschnee (kWh)", min_value=1.0, value=5.0, step=0.5)

    # Parameter sammeln
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

    # Hauptbereich: Datenverarbeitung und Anzeige
    # Date-Range erstellen
    start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
    if end_month == 12:
        end_date = pd.Timestamp(year=end_year, month=end_month, day=31)
    else:
        end_date = pd.Timestamp(year=end_year, month=end_month + 1, day=1) - pd.Timedelta(days=1)

    # Daten nach Zeitraum und Szenario filtern
    with st.spinner("Verarbeite Daten..."):
        # Neue Filterfunktion verwenden, die mit nicht-eindeutigen Zeitstempeln umgehen kann
        snow_data = filter_data_by_scenario_and_time(ds, chosen_scenario, start_date, end_date)
        df = calculate_snow_resource_data(snow_data, params)

    # Daten anzeigen wenn vorhanden
    if not df.empty:
        # Zusammenfassung
        render_summary_metrics(df, start_date, end_date)

        # Diagramme
        render_all_charts(df)

        # Divider
        st.divider()

        # Detaillierte Tabelle
        display_detailed_analysis(df)
    else:
        st.warning(
            "Keine Daten für die gewählten Parameter verfügbar. Bitte wählen Sie einen anderen Zeitraum oder ein anderes Szenario.")

if __name__ == "__main__":
    main()