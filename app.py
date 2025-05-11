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
            snow_demand_with_additive_m3 = snow_demand_m3 * (1 - params['additive_efficiency'])
            water_usage_with_additive_l = snow_demand_with_additive_m3 * params['water_per_m3']
            energy_usage_with_additive_kwh = snow_demand_with_additive_m3 * params['energy_per_m3']
            total_cost_with_additive = (water_usage_with_additive_l * params['water_cost_per_l']) + \
                                      (energy_usage_with_additive_kwh * params['energy_cost_per_kwh']) + \
                                      (snow_demand_with_additive_m3 * params['additive_cost_per_m3'])

            # Daten für diese Zeit speichern
            monthly_data.append({
                'Datum': date,
                'Jahr': date.year,
                'Monat': date.month,
                'MonatName': month_names[date.month - 1],
                'DurchschnittlicheSchneehöhe': avg_snow_depth,
                'Schneebedarf_m3': snow_demand_m3,
                'Schneebedarf_mit_Additiv_m3': snow_demand_with_additive_m3,
                'Wasserverbrauch_l': water_usage_l,
                'Wasserverbrauch_mit_Additiv_l': water_usage_with_additive_l,
                'Energieverbrauch_kwh': energy_usage_kwh,
                'Energieverbrauch_mit_Additiv_kwh': energy_usage_with_additive_kwh,
                'Gesamtkosten': total_cost,
                'Gesamtkosten_mit_Additiv': total_cost_with_additive,
                'Kosteneinsparung': total_cost - total_cost_with_additive
            })

    return pd.DataFrame(monthly_data) if monthly_data else pd.DataFrame()

def render_summary_metrics(df):
    """Zeigt die Zusammenfassungsmetriken an"""
    st.subheader("Zusammenfassung")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Gesamter Schneebedarf", f"{df['Schneebedarf_m3'].sum():.1f} m³")
        st.metric("Schneebedarf mit Additiv", f"{df['Schneebedarf_mit_Additiv_m3'].sum():.1f} m³")

    with col2:
        st.metric("Gesamtkosten ohne Additiv", f"{df['Gesamtkosten'].sum():.2f} €")
        st.metric("Gesamtkosten mit Additiv", f"{df['Gesamtkosten_mit_Additiv'].sum():.2f} €")
        st.metric("Kosteneinsparung mit Additiv", f"{df['Kosteneinsparung'].sum():.2f} €",
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
    fig.add_trace(go.Bar(
        x=df['Datum'],
        y=df['Schneebedarf_mit_Additiv_m3'],
        name='Mit Additiv'
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
        name='Ohne Additiv'
    ))
    fig.add_trace(go.Bar(
        x=df['Datum'],
        y=df['Gesamtkosten_mit_Additiv'],
        name='Mit Additiv'
    ))
    fig.update_layout(
        title="Kosten pro Monat",
        xaxis_title="Datum",
        yaxis_title="Kosten (€)",
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
            name='Ohne Additiv'
        ))
        fig.add_trace(go.Bar(
            x=df['Datum'],
            y=df['Wasserverbrauch_mit_Additiv_l'] / 1000,
            name='Mit Additiv'
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
            name='Ohne Additiv'
        ))
        fig.add_trace(go.Bar(
            x=df['Datum'],
            y=df['Energieverbrauch_mit_Additiv_kwh'],
            name='Mit Additiv'
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
                      'Schneebedarf_mit_Additiv_m3', 'Gesamtkosten', 'Gesamtkosten_mit_Additiv',
                      'Kosteneinsparung']]

    detailed_df = detailed_df.rename(columns={
        'Datum': 'Datum',
        'DurchschnittlicheSchneehöhe': 'Schneehöhe (m)',
        'Schneebedarf_m3': 'Schneebedarf (m³)',
        'Schneebedarf_mit_Additiv_m3': 'Schneebedarf mit Additiv (m³)',
        'Gesamtkosten': 'Kosten (€)',
        'Gesamtkosten_mit_Additiv': 'Kosten mit Additiv (€)',
        'Kosteneinsparung': 'Einsparung (€)'
    })

    # Tabelle anzeigen mit Formatierung
    st.dataframe(detailed_df.style.format({
        'Schneehöhe (m)': '{:.2f}',
        'Schneebedarf (m³)': '{:.1f}',
        'Schneebedarf mit Additiv (m³)': '{:.1f}',
        'Kosten (€)': '{:.2f}',
        'Kosten mit Additiv (€)': '{:.2f}',
        'Einsparung (€)': '{:.2f}'
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
    st.set_page_config(page_title="Ski Resort Schneemanagementsystem", layout="wide")

    # Seitentitel und Beschreibung
    st.title("Ski Resort Schneemanagementsystem")
    st.write("Analysieren Sie den Schneebedarf und vergleichen Sie Kosten mit und ohne Effizienzadditiv.")

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

    # Debug-Informationen zur Datenstruktur - kann später entfernt werden
    with st.expander("Debug-Informationen"):
        st.write("Dimensionen:", ds.dims)
        st.write("Koordinaten:", list(ds.coords))
        st.write("Datenstruktur:", ds.snow_depth.shape)

        # Prüfen, ob Zeitstempel eindeutig sind
        time_values = ds.time.values
        unique_times = np.unique(time_values)
        st.write(f"Zeitstempel insgesamt: {len(time_values)}")
        st.write(f"Eindeutige Zeitstempel: {len(unique_times)}")
        st.write(f"Duplikate: {len(time_values) - len(unique_times)}")

    # Sidebar für Benutzereingaben
    st.sidebar.header("Parameter")

    # Verfügbare Szenarien abrufen
    available_scenarios = list(np.unique(ds.scenario.values)) if 'scenario' in ds.coords else ["Standardszenario"]

    # Szenario-Auswahl
    chosen_scenario = st.sidebar.selectbox("Szenario wählen", available_scenarios)

    # Basiseingaben
    min_snow_depth = st.sidebar.number_input("Mindestschneehöhe für Skifahren (m)", min_value=0.1, value=0.5, step=0.1)

    # Saisondaten
    st.sidebar.subheader("Skisaison")
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

    # Hangfläche
    slope_area = st.sidebar.number_input("Hangfläche (m²)", min_value=1000, value=50000, step=1000)

    # Prognosezeitraum
    st.sidebar.subheader("Prognosezeitraum")

    # Verfügbare Zeiträume aus den Daten extrahieren
    time_min = pd.to_datetime(ds.time.min().values)
    time_max = pd.to_datetime(ds.time.max().values)

    # Jahre und Monate für Start- und Endauswahl
    start_year = st.sidebar.slider("Startjahr",
                               min_value=int(time_min.year),
                               max_value=int(time_max.year),
                               value=int(time_min.year))

    start_month = st.sidebar.slider("Startmonat",
                                min_value=1,
                                max_value=12,
                                value=1)

    end_year = st.sidebar.slider("Endjahr",
                             min_value=int(time_min.year),
                             max_value=int(time_max.year),
                             value=min(int(time_min.year) + 5, int(time_max.year)))

    end_month = st.sidebar.slider("Endmonat",
                              min_value=1,
                              max_value=12,
                              value=12)

    # Zusatzstoffeingaben
    st.sidebar.subheader("Zusatzstoffeffizienz")
    additive_efficiency = st.sidebar.slider("Zusatzstoffeffizienz (%)",
                                        min_value=5,
                                        max_value=50,
                                        value=20,
                                        step=5) / 100

    # Kosten- und Ressourcenparameter
    st.sidebar.subheader("Kosten- und Ressourcenparameter")
    additive_cost_per_m3 = st.sidebar.number_input("Zusatzstoffkosten pro m³ (€)", min_value=0.1, value=2.0, step=0.1)
    water_per_m3 = st.sidebar.number_input("Wasserverbrauch pro m³ Kunstschnee (l)", min_value=50, value=200, step=10)
    energy_per_m3 = st.sidebar.number_input("Energieverbrauch pro m³ Kunstschnee (kWh)", min_value=1.0, value=5.0,
                                      step=0.5)
    water_cost_per_l = st.sidebar.number_input("Wasserkosten pro Liter (€)", min_value=0.0001, value=0.002, step=0.0005,
                                          format="%.4f")
    energy_cost_per_kwh = st.sidebar.number_input("Energiekosten pro kWh (€)", min_value=0.01, value=0.25, step=0.01)

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
        render_summary_metrics(df)

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