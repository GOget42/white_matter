import streamlit as st
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Seitentitel und Beschreibung
st.title("Ski Resort Schneemanagementsystem")
st.write("Analysieren Sie den Schneebedarf und vergleichen Sie Kosten mit und ohne Effizienzadditiv.")

# Sidebar für Benutzereingaben
st.sidebar.header("Parameter")

# NetCDF-Dateipfad
nc_file = st.sidebar.file_uploader("NetCDF-Datei hochladen (.nc)", type=["nc"])

if nc_file:
    # Datei in temporäres Verzeichnis speichern
    with open("temp_snow_data.nc", "wb") as f:
        f.write(nc_file.getbuffer())

    # Daten laden
    ds = xr.open_dataset("temp_snow_data.nc")

    # Verfügbare Szenarien abrufen
    available_scenarios = list(np.unique(ds.scenario.values)) if 'scenario' in ds else ["Standardszenario"]

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

    # Szenario-Auswahl
    chosen_scenario = st.sidebar.selectbox("Szenario wählen", available_scenarios)

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

    # Hauptbereich: Datenverarbeitung und Anzeige
    st.header("Schneebedarfsanalyse")

    # Date-Range erstellen
    start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
    if end_month == 12:
        end_date = pd.Timestamp(year=end_year, month=end_month, day=31)
    else:
        end_date = pd.Timestamp(year=end_year, month=end_month + 1, day=1) - pd.Timedelta(days=1)

    # Daten nach Zeitraum und Szenario filtern
    snow_data = ds.snow_depth.sel(time=slice(start_date, end_date))

    if 'scenario' in ds:
        snow_data = snow_data.where(snow_data.scenario == chosen_scenario, drop=True)

    # Berechnung des durchschnittlichen Schneebedarfs pro Monat
    monthly_data = []

    # Monate durchlaufen und Schneebedarf berechnen
    for time_point in snow_data.time.values:
        date = pd.to_datetime(time_point)

        # Nur Saisonmonate berücksichtigen
        if season_start_month <= season_end_month:
            is_in_season = (date.month >= season_start_month) and (date.month <= season_end_month)
        else:
            # Bei saisonübergreifenden Zeiträumen (z.B. Nov-März)
            is_in_season = (date.month >= season_start_month) or (date.month <= season_end_month)

        if is_in_season:
            # Durchschnittliche Schneehöhe für diesen Zeitpunkt
            avg_snow_depth = float(snow_data.sel(time=time_point).mean(['latitude', 'longitude']).values)

            # Schneebedarf berechnen
            snow_demand_m3 = max(0, (min_snow_depth - avg_snow_depth) * slope_area)

            # Ressourcenberechnung ohne Zusatzstoff
            water_usage_l = snow_demand_m3 * water_per_m3
            energy_usage_kwh = snow_demand_m3 * energy_per_m3
            total_cost = (water_usage_l * water_cost_per_l) + (energy_usage_kwh * energy_cost_per_kwh)

            # Ressourcenberechnung mit Zusatzstoff
            snow_demand_with_additive_m3 = snow_demand_m3 * (1 - additive_efficiency)
            water_usage_with_additive_l = snow_demand_with_additive_m3 * water_per_m3
            energy_usage_with_additive_kwh = snow_demand_with_additive_m3 * energy_per_m3
            total_cost_with_additive = (water_usage_with_additive_l * water_cost_per_l) + \
                                       (energy_usage_with_additive_kwh * energy_cost_per_kwh) + \
                                       (snow_demand_with_additive_m3 * additive_cost_per_m3)

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

    # DataFrame erstellen
    df = pd.DataFrame(monthly_data)

    # Daten anzeigen wenn vorhanden
    if len(df) > 0:
        # Zusammenfassung
        st.subheader("Zusammenfassung")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Gesamter Schneebedarf", f"{df['Schneebedarf_m3'].sum():.1f} m³")
            st.metric("Schneebedarf mit Additiv", f"{df['Schneebedarf_mit_Additiv_m3'].sum():.1f} m³")

        with col2:
            st.metric("Gesamtkosten ohne Additiv", f"{df['Gesamtkosten'].sum():.2f} €")
            st.metric("Gesamtkosten mit Additiv", f"{df['Gesamtkosten_mit_Additiv'].sum():.2f} €")
            st.metric("Kosteneinsparung mit Additiv", f"{df['Kosteneinsparung'].sum():.2f} €",
                      delta=f"{df['Kosteneinsparung'].sum() / df['Gesamtkosten'].sum() * 100:.1f}%")

        # Monatlicher Schneebedarf mit Plotly
        st.subheader("Monatlicher Schneebedarf")

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=df['Datum'],
            y=df['Schneebedarf_m3'],
            name='Standard'
        ))
        fig1.add_trace(go.Bar(
            x=df['Datum'],
            y=df['Schneebedarf_mit_Additiv_m3'],
            name='Mit Additiv'
        ))
        fig1.update_layout(
            title="Schneebedarf pro Monat",
            xaxis_title="Datum",
            yaxis_title="Schneebedarf (m³)",
            barmode='group'
        )
        st.plotly_chart(fig1)

        # Kosten
        st.subheader("Kostenvergleich")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df['Datum'],
            y=df['Gesamtkosten'],
            name='Ohne Additiv'
        ))
        fig2.add_trace(go.Bar(
            x=df['Datum'],
            y=df['Gesamtkosten_mit_Additiv'],
            name='Mit Additiv'
        ))
        fig2.update_layout(
            title="Kosten pro Monat",
            xaxis_title="Datum",
            yaxis_title="Kosten (€)",
            barmode='group'
        )
        st.plotly_chart(fig2)

        # Ressourcenverbrauch
        st.subheader("Ressourcenverbrauch")
        resource_choice = st.radio(
            "Ressourcenvergleich anzeigen für:",
            ("Wasserverbrauch", "Energieverbrauch")
        )

        if resource_choice == "Wasserverbrauch":
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=df['Datum'],
                y=df['Wasserverbrauch_l'] / 1000,  # Umrechnung in m³
                name='Ohne Additiv'
            ))
            fig3.add_trace(go.Bar(
                x=df['Datum'],
                y=df['Wasserverbrauch_mit_Additiv_l'] / 1000,
                name='Mit Additiv'
            ))
            fig3.update_layout(
                title="Wasserverbrauch pro Monat",
                xaxis_title="Datum",
                yaxis_title="Wasserverbrauch (m³)",
                barmode='group'
            )
            st.plotly_chart(fig3)
        else:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=df['Datum'],
                y=df['Energieverbrauch_kwh'],
                name='Ohne Additiv'
            ))
            fig3.add_trace(go.Bar(
                x=df['Datum'],
                y=df['Energieverbrauch_mit_Additiv_kwh'],
                name='Mit Additiv'
            ))
            fig3.update_layout(
                title="Energieverbrauch pro Monat",
                xaxis_title="Datum",
                yaxis_title="Energieverbrauch (kWh)",
                barmode='group'
            )
            st.plotly_chart(fig3)

        # Detaillierte Tabelle
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

        # Tabelle anzeigen mit formatierung
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
    else:
        st.warning(
            "Keine Daten für die gewählten Parameter verfügbar. Bitte wählen Sie einen anderen Zeitraum oder ein anderes Szenario.")

else:
    st.info("Bitte laden Sie eine NetCDF-Datei (.nc) mit Schneehöhendaten hoch, um zu beginnen.")
    st.write("Die Datei sollte eine Variable 'snow_depth' mit Abmessungen für Zeit, Breiten- und Längengrad enthalten.")