import streamlit as st
import pandas as pd
import numpy as np

st.markdown(
    """
    <style>
    .titlu {
        color: #FFFFFF;
        font-size: 50px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

section = st.sidebar.radio("Navigați la:",
                           ["Proiect", "Informații"])

#Citire set de date
df = pd.read_csv('data/Samsung.csv')
df['Date'] = pd.to_datetime(df['Date']).dt.date #citirea coloanei cauzeaza o problema de conversie


if section == 'Proiect':
    st.markdown('<h1 class="titlu">Proiect PSW</h1>', unsafe_allow_html=True)

    sub_section = st.sidebar.radio("Secțiuni din proiect", ["Prezentare date", "Filtrări pe baza datelor"])

    if sub_section == "Prezentare date":
        st.markdown('## Prezentare date')
        st.markdown('#### Evoluția prețului acțiunilor Samsung Electronics')
        st.dataframe(df)

        st.markdown('#### Descrierea setului de date')
        st.markdown('##### Setul de date furnizează informații cu privire la evoluția zilnică a prețului acțiunilor Samsung Electronics.')
        st.markdown('##### Prețurile sunt exprimate în KRW (won sub-coreean).')
        st.info("**`Date`** → Data sesiunii de tranzacționare.")
        st.info("**`Open`** → Prețul acțiunilor la începutul sesiunii (preț de deschidere).")
        st.info("**`High`** → Cel mai mare preț atins de acțiuni în timpul sesiunii.")
        st.info("**`Low`** → Cel mai mic preț atins de acțiuni în timpul sesiunii.")
        st.info("**`Close`** → Prețul acțiunilor la finalul sesiunii (preț de închidere).")
        st.info("**`Adj Close`** → Preț de închidere ajustat (ia în considerare acțiuni corporative sau alte ajustări).")
        st.info("**`Volume`** → Număr total de acțiuni tranzacționate.")
        st.markdown('##### Informații despre setul de date.')
        st.write('Tipuri de date:')
        st.write(df.dtypes)
        st.write(f"Dimensiunea setului de date: setul conține {df.shape[0]} rânduri și {df.shape[1]} coloane")
        st.markdown('##### Statistica descriptivă a setului de date.')
        st.dataframe(df.describe())




    elif sub_section == "Filtrări pe baza datelor":
        st.markdown('## Exemple de filtrări efectuate asupra datelor')

        st.markdown('### Filtrare #1')
        coloane1 = df.columns.tolist()
        coloane1.pop(0)
        coloane1.pop(-1)
        col_select = st.multiselect("Selectează coloanele", coloane1)
        col_select = ['Date','Volume'] + col_select
        df_filtrat = df[col_select]

        min_vol = st.slider("Afișează doar sesiunile cu volumul minim:", int(df["Volume"].min()), int(df["Volume"].max()),
                            int(round(df["Volume"].max() + df["Volume"].min()) / 2))
        df_filtrat = df_filtrat[df_filtrat["Volume"] >= min_vol]

        st.dataframe(df_filtrat)

        st.markdown('### Filtrare #2')
        start_row, end_row = st.slider("Afișează următoarele rânduri:", 0, len(df) - 1, (0, len(df) -1))
        df_filtrat1 = df.iloc[start_row:end_row + 1]
        st.dataframe(df_filtrat1)

        df_filtrat2 = df
        st.markdown('### Filtrare #3')
        sesiuni_selectate = st.multiselect("Selectează sesiunile:", df["Date"].unique().tolist())
        if sesiuni_selectate:
            df_filtrat2 = df_filtrat2.loc[df_filtrat2["Date"].isin(sesiuni_selectate)]
        st.dataframe(df_filtrat2)

        st.markdown('### Filtrare #4')
        df_filtrat3 = df
        min_data = df['Date'].min()
        max_data = df['Date'].max()
        data_range = st.date_input('Selectează intervalul de sesiuni:', [min_data,max_data],min_value=min_data,max_value=max_data)
        if len(data_range) == 2:
            start_data,end_data = data_range
            df_filtrat3 =df_filtrat3[(df['Date'] >= start_data) & (df['Date'] <= end_data)]
            st.dataframe(df_filtrat3)



elif section == 'Informații':
    st.markdown("""
    ## 📂 Date utilizate în proiect

    📌 [Acțiuni Samsung Electronics](https://www.kaggle.com/datasets/ranugadisansagamage/samsung-stocks)    
    """)

    st.markdown("""
    ##  Proiect realizat de:
    👨‍💻 **Raicea David-Gabriel**  
    👨‍💻 **Rădulescu Theodor**  
    """)




