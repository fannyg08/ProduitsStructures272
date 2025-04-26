# dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
from structuration.ClassMarket import MarketData
from structuration.ClassVolatility import SSVIModel
from base.ClassRate import VasicekModel
from structuration.ClassPricing import PricingEngine
from structuration.Produits.ProtectedCapital import CapitalProtectedNote
from base.ClassMaturity import Maturity

# --------- Interface utilisateur Streamlit ---------

st.title("Dashboard Pricing - Produits Structurés")

st.header("Entrée des données de marché")

with st.form("formulaire_pricing"):
    # Spot
    spot = st.number_input("Spot", value=100.0)

    # Strikes
    strikes_input = st.text_input("Liste des strikes (séparés par des virgules)", "80,90,100,110,120")
    strikes = np.array([float(x.strip()) for x in strikes_input.split(",")])

    # Maturités
    maturities_input = st.text_input("Liste des maturités (en années, séparées par des virgules)", "0.5,1.0,2.0")
    maturities = np.array([float(x.strip()) for x in maturities_input.split(",")])

    # Prix du marché
    st.write("Entrez les prix de marché sous forme de tableau : une ligne par maturité, séparée par des ';', et chaque valeur séparée par une virgule.")
    market_prices_input = st.text_area("Prix de marché", "10,11,12,13,14;9,10,11,12,13;8,9,10,11,12")
    market_prices = np.array([
        [float(price) for price in line.split(",")]
        for line in market_prices_input.strip().split(";")
    ])

    # Vérification dimensions
    if market_prices.shape != (len(maturities), len(strikes)):
        st.error("Le tableau de prix doit correspondre aux strikes et maturités.")
        st.stop()

    # Option type
    option_type = st.selectbox("Type d'option", ["call", "put"])

    # Taux sans risque
    risk_free_rate = st.number_input("Taux sans risque (ex: 0.03 pour 3%)", value=0.03)

    # Dividende
    dividend_yield = st.number_input("Taux de dividende (ex: 0.00)", value=0.0)

    st.header("Paramètres du produit structuré")

    underlying_id = st.text_input("Nom du sous-jacent (ex: SPX)", "SPX")
    maturity_years = st.slider("Maturité (en années)", min_value=0.5, max_value=10.0, value=5.0, step=0.5)
    strike = st.number_input("Strike produit", value=100.0)
    participation_rate = st.number_input("Taux de participation", value=1.5)
    capital_protection = st.slider("Protection du capital", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
    nominal = st.number_input("Nominal", value=1000.0)

    # BOUTON
    submitted = st.form_submit_button("Lancer le pricing")

if submitted:
    # --------- Construction du marché ---------
    market_data = MarketData(
        spot=spot,
        maturities=maturities,
        strikes=strikes,
        market_prices=market_prices,
        option_type=option_type,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield
    )

    ssvi_model = SSVIModel(market_data=market_data)
    params = ssvi_model.calibrate()

    vasicek_model = VasicekModel(a=0.1, b=0.03, sigma=0.01, r0=risk_free_rate)

    engine = PricingEngine(
        spot_price=spot,
        domestic_rate=vasicek_model,
        volatility=ssvi_model,
        dividend=dividend_yield,
        num_paths=10000,
        num_steps=252,
        seed=42
    )

    # --------- Construction du produit ---------
    note = CapitalProtectedNote(
        underlying_id=underlying_id,
        maturity=Maturity(maturity_years),
        nominal=nominal,
        strike=strike,
        participation_rate=participation_rate,
        capital_protection=capital_protection,
        rate_model=vasicek_model,
        spot_price=spot,
        volatility_model=ssvi_model,
        dividend=dividend_yield
    )

    # Pricing
    price = engine.price(note, method="decomposition")
    greeks = engine.calculate_greeks(note, method="decomposition")

    # Résultats
    st.success(f"Prix : {price:.2f} €")

    greeks_df = pd.DataFrame.from_dict(greeks, orient='index', columns=["Valeur"])
    st.subheader("Grecques :")
    st.dataframe(greeks_df)

    # Historique
    if "historique" not in st.session_state:
        st.session_state.historique = []

    st.session_state.historique.append({
        "Sous-jacent": underlying_id,
        "Prix (€)": price,
        "Maturité (ans)": maturity_years,
        "Strike": strike,
        "Participation": participation_rate,
        "Capital Protection": capital_protection,
        **greeks
    })

# --------- Historique ---------

if "historique" in st.session_state and len(st.session_state.historique) > 0:
    st.subheader("Historique des pricings")
    df = pd.DataFrame(st.session_state.historique)
    st.dataframe(df)
    st.download_button("Télécharger l'historique en CSV", df.to_csv(index=False), file_name="historique_pricing.csv")
