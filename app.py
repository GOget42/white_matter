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
    st.subheader(f"Zusammenfassung für den Zeitraum {start_date.strftime('%m.%Y')} bis {end_date.strftime('%m.%Y')}")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Gesamter Schneebedarf", f"{df['Schneebedarf_m3'].sum():,.1f}".replace(",", "'") + " m³")

    with col2:
        st.metric("Wasserverbrauch ohne Keimbildner", f"{df['Wasserverbrauch_l'].sum() / 1000:,.1f}".replace(",", "'") + " m³")
        st.metric("Wasserverbrauch mit Keimbildner", f"{df['Wasserverbrauch_mit_Additiv_l'].sum() / 1000:,.1f}".replace(",", "'") + " m³")
        st.metric("Wassereinsparung",
                  f"{(df['Wasserverbrauch_l'].sum() - df['Wasserverbrauch_mit_Additiv_l'].sum()) / 1000:,.1f}".replace(",", "'") + " m³",
                  delta=f"{(df['Wasserverbrauch_l'].sum() - df['Wasserverbrauch_mit_Additiv_l'].sum()) / df['Wasserverbrauch_l'].sum() * 100:.1f}%"
                    if df['Wasserverbrauch_l'].sum() > 0 else None)

    with col3:
        st.metric("Energieverbrauch ohne Keimbildner", f"{df['Energieverbrauch_kwh'].sum():,.1f}".replace(",", "'") + " kWh")
        st.metric("Energieverbrauch mit Keimbildner", f"{df['Energieverbrauch_mit_Additiv_kwh'].sum():,.1f}".replace(",", "'") + " kWh")
        st.metric("Energieeinsparung",
                  f"{df['Energieverbrauch_kwh'].sum() - df['Energieverbrauch_mit_Additiv_kwh'].sum():,.1f}".replace(",", "'") + " kWh",
                  delta=f"{(df['Energieverbrauch_kwh'].sum() - df['Energieverbrauch_mit_Additiv_kwh'].sum()) / df['Energieverbrauch_kwh'].sum() * 100:.1f}%"
                    if df['Energieverbrauch_kwh'].sum() > 0 else None)

    with col4:
        st.metric("Gesamtkosten ohne Keimbildner", f"{df['Gesamtkosten'].sum():,.2f}".replace(",", "'") + " CHF")
        st.metric("Gesamtkosten mit Keimbildner", f"{df['Gesamtkosten_mit_Additiv'].sum():,.2f}".replace(",", "'") + " CHF")
        st.metric("Kosteneinsparung",
                  f"{df['Kosteneinsparung'].sum():,.2f}".replace(",", "'") + " CHF",
                  delta=f"{df['Kosteneinsparung'].sum() / df['Gesamtkosten'].sum() * 100:.1f}%"
                  if df['Gesamtkosten'].sum() > 0 else None)

def plot_snow_demand(df):
    """Erstellt ein Diagramm des monatlichen Schneebedarfs"""
    st.subheader("Monatlicher Schneebedarf")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Datum'],
        y=df['Schneebedarf_m3'],
        name='Standard'
    ))
    fig.update_layout(
        title="Schneebedarf pro Monat",
        xaxis_title="Datum",
        yaxis_title="Schneebedarf (m³)",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig)

def plot_costs(df):
    """Erstellt ein Diagramm der monatlichen Kosten"""
    st.subheader("Kostenvergleich")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Datum'],
        y=df['Gesamtkosten'],
        name='Ohne Keimbildner'
    ))
    fig.add_trace(go.Bar(
        x=df['Datum'],
        y=df['Gesamtkosten_mit_Additiv'],
        name='Mit Keimbildner'
    ))
    fig.update_layout(
        title="Kosten pro Monat",
        xaxis_title="Datum",
        yaxis_title="Kosten (CHF)",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig)

def plot_resource_usage(df, resource_choice):
    """Erstellt ein Diagramm des Ressourcenverbrauchs"""
    st.subheader("Ressourcenverbrauch")

    fig = go.Figure()

    if resource_choice == "Wasserverbrauch":
        fig.add_trace(go.Bar(
            x=df['Datum'],
            y=df['Wasserverbrauch_l'] / 1000,  # Umrechnung in m³
            name='Ohne Keimbildner'
        ))
        fig.add_trace(go.Bar(
            x=df['Datum'],
            y=df['Wasserverbrauch_mit_Additiv_l'] / 1000,
            name='Mit Keimbildner'
        ))
        fig.update_layout(
            title="Wasserverbrauch pro Monat",
            xaxis_title="Datum",
            yaxis_title="Wasserverbrauch (m³)",
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else:
        fig.add_trace(go.Bar(
            x=df['Datum'],
            y=df['Energieverbrauch_kwh'],
            name='Ohne Keimbildner'
        ))
        fig.add_trace(go.Bar(
            x=df['Datum'],
            y=df['Energieverbrauch_mit_Additiv_kwh'],
            name='Mit Keimbildner'
        ))
        fig.update_layout(
            title="Energieverbrauch pro Monat",
            xaxis_title="Datum",
            yaxis_title="Energieverbrauch (kWh)",
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

    st.plotly_chart(fig)

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
    slope_area = st.sidebar.number_input("Hangfläche (m²)", min_value=1000, value=1000000, step=10000)

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
    st.header("Schneebedarfsanalyse")

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

        # Monatlicher Schneebedarf
        plot_snow_demand(df)

        # Kosten
        plot_costs(df)

        # Ressourcenverbrauch
        resource_choice = st.radio(
            "Ressourcenvergleich anzeigen für:",
            ("Wasserverbrauch", "Energieverbrauch")
        )
        plot_resource_usage(df, resource_choice)

        # Detaillierte Tabelle
        display_detailed_analysis(df)
    else:
        st.warning(
            "Keine Daten für die gewählten Parameter verfügbar. Bitte wählen Sie einen anderen Zeitraum oder ein anderes Szenario.")

if __name__ == "__main__":
    main()