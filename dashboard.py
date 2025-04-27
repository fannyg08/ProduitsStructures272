# dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime 

from structuration.ClassMarket import MarketData
from structuration.ClassVolatility import SSVIModel
from base.ClassRate import VasicekModel
from structuration.ClassPricing import PricingEngine
from structuration.Produits.ProtectedCapital import CapitalProtectedNote
from base.ClassMaturity import Maturity

from structuration.Produits.Complex import AthenaProduct
from structuration.Produits.Complex import PhoenixProduct
from structuration.Produits.Complex import RangeAccrualNote


# --------- Interface utilisateur Streamlit ---------

st.set_page_config(
    page_title="Dashboard Pricing Produits Structurés",
    page_icon=":moneybag:",
    layout="wide"
)

col_market, col_product, col_result = st.columns(3)

with col_market:
    st.header("Données de Marché")
    
    spot = st.number_input("Spot", value=100.0)

    # Strikes : nombre et bornes
    st.markdown("##### Strikes")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_strikes = st.number_input("Nombre", min_value=1, max_value=20, value=5, step=1)
    with col2:
        strike_min = st.number_input("Min.",min_value=0.0, max_value=200.0,  value=80.0, step=1.0)
    with col3:
        strike_max = st.number_input("Max.", min_value=0.0, max_value=200.0, value=120.0, step=1.0)
    
    strikes = np.linspace(strike_min, strike_max, int(n_strikes))

    # Maturities : nombre et bornes
    st.markdown("##### Maturités (en années)")
    # Nombre de maturités et bornes
    col1, col2, col3 = st.columns(3)
    with col1:
        n_maturities = st.number_input("Nombre", min_value=1, max_value=20, value=3, step=1)
    with col2:
        maturity_min = st.number_input("Min.",min_value=0.1, max_value=100.0, value=0.5, step=1.0)
    with col3:
        maturity_max = st.number_input("Max.", min_value=0.1, max_value=100.0, value=2.0, step=1.0)
    
    maturities = np.linspace(maturity_min, maturity_max, int(n_maturities))

    # Prix de Marché dans un tableau éditable
    st.markdown("##### Prix de Marché (éditables)")

    # Créer un tableau par défaut aléatoire
    default_prices = np.round(10 + np.random.randn(len(maturities), len(strikes)), 2)
    default_df = pd.DataFrame(
        default_prices, 
        index=[f"{t:.2f}y" for t in maturities], 
        columns=[f"{k:.2f}" for k in strikes]
    )

    # Editeur interactif
    market_prices_df = st.data_editor(
        default_df,
        num_rows="fixed",
        use_container_width=True
    )

    market_prices = market_prices_df.values

    if market_prices.shape != (len(maturities), len(strikes)):
        st.error("Erreur : dimensions des prix incohérentes avec strikes/maturités.")
        st.stop()

    # Taux côte à côte
    col1, col2 = st.columns(2)

    with col1:
        risk_free_rate = st.number_input("Taux sans risque", value=0.03, step=0.001, format="%.4f")
    
    with col2:
        dividend_yield = st.number_input("Taux de dividende", value=0.00, step=0.001, format="%.4f")

    # Type d'option
    option_type = st.selectbox("Type d'Option", ["call", "put"])
    pricing_date = st.date_input("Date de Pricing", value=datetime.today())



# --------- Colonne 2 : Produit ---------
with col_product:
    st.header("Paramètres du Produit")

    product_choice = st.selectbox(
        "Sélectionnez un produit",
        [
            "Capital Protected Note",
            "Athena",
            "Phoenix",
            "Range Accrual Note",
            "Reverse Convertible",
            "Discount Certificate"
        ],
        index=0
    )

    # Interface Produit
    underlying_id = st.text_input("Sous-jacent", "SPX")
    maturity_years = st.slider("Maturité (années)", 0.5, 30.0, 5.0, step=0.5)
    nominal = st.number_input("Nominal (€)", value=1000.0)

    if product_choice == "Capital Protected Note":
        strike = st.number_input("Strike du produit", value=100.0)
        participation_rate = st.number_input("Taux de participation", value=1.5)
        capital_protection = st.slider("Protection du capital (%)", 0.0, 1.0, 0.9, step=0.05)

    if product_choice in ["Athena", "Phoenix"]:
        observation_dates = st.text_area("Dates d'observation (YYYY-MM-DD, séparées par ',')", "2025-12-31,2026-12-31,2027-12-31")
        autocall_barriers = st.text_input("Barrières d'autocall (%)", "1.0,1.0,1.0")
        coupon_barriers = st.text_input("Barrières de coupon (%)", "0.7,0.7,0.7")
        coupons = st.text_input("Coupons (%)", "0.05,0.05,0.05")
        capital_barrier = st.number_input("Barrière de protection capital (%)", value=0.6)
        memory_effect = st.checkbox("Effet mémoire", value=True)

    if product_choice == "Range Accrual Note":
        coupon_rate = st.number_input("Taux de coupon annuel (%)", value=5.0) / 100
        lower_barrier = st.number_input("Barrière basse (% du spot)", value=0.8)
        upper_barrier = st.number_input("Barrière haute (% du spot)", value=1.2)
        observation_dates = st.text_area("Dates d'observation (YYYY-MM-DD, séparées par ',')", "2025-06-30,2025-12-31,2026-06-30,2026-12-31")
        payment_dates = st.text_area("Dates de paiement (YYYY-MM-DD, séparées par ',')", "2025-12-31,2026-12-31")
        capital_protection = st.slider("Protection capital (%)", 0.0, 1.0, 1.0, step=0.05)

    if product_choice == "Reverse Convertible":
        coupon = st.number_input("Coupon (%)", value=5.0)
        strike_level = st.number_input("Niveau de strike (% du spot)", value=100.0)

    if product_choice == "Discount Certificate":
        discount = st.number_input("Décote (%)", value=10.0)
        cap_level = st.number_input("Cap (% du spot)", value=110.0)

    submitted = st.button("Lancer le Pricing", use_container_width=True)

# --------- Colonne 3 : Résultats ---------
with col_result:
    st.header("Résultats")

    if submitted:
        # --- Marché
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
        ssvi_model.calibrate()

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

        # --- Produit
        note = None

        if product_choice == "Capital Protected Note":
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

        elif product_choice == "Athena":
            note = AthenaProduct(
                underlying_id=underlying_id,
                maturity=Maturity(maturity_years),
                observation_dates=[datetime.strptime(d.strip(), "%Y-%m-%d").date() for d in observation_dates.split(",")],
                autocall_barriers=[float(x.strip()) for x in autocall_barriers.split(",")],
                coupon_barriers=[float(x.strip()) for x in coupon_barriers.split(",")],
                coupons=[float(x.strip()) for x in coupons.split(",")],
                capital_barrier=capital_barrier,
                memory_effect=memory_effect,
                nominal=nominal, 
                spot_price=spot,
                pricing_date=pricing_date
            )

        elif product_choice == "Phoenix":
            note = PhoenixProduct(
                underlying_id=underlying_id,
                maturity=Maturity(maturity_years),
                observation_dates=[datetime.strptime(d.strip(), "%Y-%m-%d").date() for d in observation_dates.split(",")],
                autocall_barriers=[float(x.strip()) for x in autocall_barriers.split(",")],
                coupon_barriers=[float(x.strip()) for x in coupon_barriers.split(",")],
                coupons=[float(x.strip()) for x in coupons.split(",")],
                capital_barrier=capital_barrier,
                memory_effect=memory_effect,
                nominal=nominal
            )

        elif product_choice == "Range Accrual Note":
            note = RangeAccrualNote(
                underlying_id=underlying_id,
                maturity=Maturity(maturity_years),
                coupon_rate=coupon_rate,
                lower_barrier=lower_barrier,
                upper_barrier=upper_barrier,
                observation_dates=[datetime.strptime(d.strip(), "%Y-%m-%d").date() for d in observation_dates.split(",")],
                payment_dates=[datetime.strptime(d.strip(), "%Y-%m-%d").date() for d in payment_dates.split(",")],
                capital_protection=capital_protection,
                nominal=nominal
            )

        elif product_choice == "Reverse Convertible":
            from structuration.Produits.YieldEnhancement import ReverseConvertible
            note = ReverseConvertible(
                underlying_id=underlying_id,
                maturity=Maturity(maturity_years),
                nominal=nominal,
                coupon=coupon / 100,
                strike_level=strike_level / 100
            )

        elif product_choice == "Discount Certificate":
            from structuration.Produits.YieldEnhancement import DiscountCertificate
            note = DiscountCertificate(
                underlying_id=underlying_id,
                maturity=Maturity(maturity_years),
                nominal=nominal,
                discount=discount / 100,
                cap_level=cap_level / 100
            )

        # --- Pricing
        price = engine.price(note, method="decomposition")
        greeks = engine.calculate_greeks(note, method="decomposition")

        st.success(f"💰 Prix estimé : {price:.2f} €")

        greeks_df = pd.DataFrame.from_dict(greeks, orient='index', columns=["Valeur"])
        st.dataframe(greeks_df, use_container_width=True)

        # --- Historique
        if "historique" not in st.session_state:
            st.session_state.historique = []

        historique = {
            "Produit": product_choice,
            "Sous-jacent": underlying_id,
            "Prix (€)": price,
            "Maturité (ans)": maturity_years,
            "Nominal (€)": nominal,
            **greeks
        }
        st.session_state.historique.append(historique)

        # --- Surface de volatilité
        st.subheader("Surface de Volatilité Implicite")

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(strikes, maturities)
        Z = np.zeros_like(X)

        for i, t in enumerate(maturities):
            for j, k in enumerate(strikes):
                Z[i, j] = ssvi_model.get_implied_volatility(k, t)

        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel("Strike")
        ax.set_ylabel("Maturité")
        ax.set_zlabel("Volatilité")
        st.pyplot(fig)

# --------- Historique global ---------
st.divider()

if "historique" in st.session_state and len(st.session_state.historique) > 0:
    st.subheader("Historique des Pricings")
    df = pd.DataFrame(st.session_state.historique)
    st.dataframe(df, use_container_width=True)
    st.download_button("Télécharger l'historique CSV", df.to_csv(index=False), file_name="historique_pricing.csv", use_container_width=True)

if __name__ == "__main__":
    import os
    if not any("streamlit" in arg for arg in os.sys.argv):
        os.system(f"streamlit run {__file__}")