import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

# Citire și conversie corectă pentru compatibilitate completă
df = pd.read_csv('data/Samsung.csv')
df['Date'] = pd.to_datetime(df['Date']).dt.date
df_final = df.copy()
df_scaled_model = df.copy()
df_final = df_final[df_final["Volume"] > 30000]

# 1. Tratarea outlierilor (mediana)

df_numeric = df_final.select_dtypes(include=[np.number])
z_scores = np.abs(zscore(df_numeric))
extreme_mask = (z_scores > 3)
mediane = df_numeric.median()
for col in df_numeric.columns:
    mask_col = extreme_mask[:, df_numeric.columns.get_loc(col)]
    df_final.loc[mask_col, col] = mediane[col]

# 3. one hot
df_final["Luna"] = pd.to_datetime(df_final["Date"]).dt.month
df_final["Anotimp"] = df_final["Luna"].apply(lambda luna:
    "Iarna" if luna in [12, 1, 2] else
    "Primavara" if luna in [3, 4, 5] else
    "Vara" if luna in [6, 7, 8] else "Toamna"
)
df_final = pd.get_dummies(df_final, columns=["Anotimp"])

# 4. Standard scaler
scaler_model = StandardScaler()
scaled_values_model = scaler_model.fit_transform(df_final.select_dtypes(include=[np.number]))
df_scaled_model = df_final.copy()
df_scaled_model[[f"{col}_scaled" for col in df_final.select_dtypes(include=[np.number]).columns]] = scaled_values_model




if section == 'Proiect':
    st.markdown('<h1 class="titlu">Proiect PSW</h1>', unsafe_allow_html=True)

    sub_section = st.sidebar.radio("Secțiuni din proiect", ["Prezentare date", "Filtrări pe baza datelor", "Tratarea valorilor lipsă și a valorilor extreme", "Metode de codificare a datelor", "Analiza corelațiilor","Metode de scalare a datelor", "Prelucrari statistice și agregare" ])

    if sub_section == "Prezentare date":
        st.markdown('## Prezentare date')
        st.markdown('#### Evoluția prețului acțiunilor Samsung Electronics')
        st.dataframe(df)

        st.markdown('#### Descrierea setului de date')
        st.markdown('##### Setul de date furnizează informații cu privire la evoluția zilnică a prețului acțiunilor Samsung Electronics.')
        st.markdown('##### Prețurile sunt exprimate în KRW (won sud-coreean).')
        st.info("**`Date`** → Data sesiunii de tranzacționare.")
        st.info("**`Open`** → Prețul acțiunilor la începutul sesiunii (preț de deschidere).")
        st.info("**`High`** → Cel mai mare preț atins de acțiuni în timpul sesiunii.")
        st.info("**`Low`** → Cel mai mic preț atins de acțiuni în timpul sesiunii.")
        st.info("**`Close`** → Prețul acțiunilor la finalul sesiunii (preț de închidere).")
        st.info("**`Adj Close`** → Preț de închidere ajustat (ia în considerare acțiuni corporative sau alte ajustări).")
        st.info("**`Volume`** → Număr total de acțiuni tranzacționate.")
        st.markdown('##### Informații despre setul de date.')
        st.write('Tipuri de date:')
        st.write({col: str(dtype) for col, dtype in df.dtypes.items()}) #reparat eroare principala
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



    elif sub_section == "Tratarea valorilor lipsă și a valorilor extreme":

        st.markdown('## Tratarea valorilor lipsă')

        st.markdown(
            '### Întrucât setul nostru de date nu are valori lipsă, simulăm acest aspect eliminând câteva valori.')

        # simulare lipsuri

        df_simulat = df.copy()

        valori_lipsa_index = [5, 10, 15, 20, 25]
        coloane_afectate = ['Open', 'High', 'Close', 'Adj Close', 'Volume']

        for idx, col in zip(valori_lipsa_index, coloane_afectate):
            df_simulat.loc[idx, col] = np.nan

        metoda = st.radio("Alege metoda de tratare a valorilor lipsă:", ["Eliminare (dropna)", "Completare (fillna)"])

        if metoda == "Eliminare (dropna)":

            df_tratat = df_simulat.dropna()
            st.success(f"Setul rezultat are {df_tratat.shape[0]} rânduri (după eliminare).")
            st.markdown("##### Rândurile eliminate:")
            st.dataframe(df_simulat.loc[valori_lipsa_index])

        else:

            metoda_fill = st.selectbox("Alege metoda de completare:", ["Medie", "Mediană", "Zero"])
            df_tratat = df_simulat.copy()

            if metoda_fill == "Medie":
                df_tratat.fillna(df_tratat.mean(numeric_only=True), inplace=True)

            elif metoda_fill == "Mediană":
                df_tratat.fillna(df_tratat.median(numeric_only=True), inplace=True)

            else:
                df_tratat.fillna(0, inplace=True)

            st.success("Valorile lipsă au fost completate.")
            st.markdown("##### Rândurile modificate (colorăm valorile completate):")


            def colorize(val, col, idx):

                return 'background-color: green' if pd.isna(df_simulat.loc[idx, col]) else ''

            styled = df_tratat.loc[valori_lipsa_index].style.apply(
                lambda row: [colorize(row[col], col, row.name) for col in row.index], axis=1
            )
            st.dataframe(styled, use_container_width=True)

        st.markdown('---')
        st.markdown('## Tratarea valorilor extreme')
        st.markdown('### Detectarea valorilor extreme pe baza Z-score (> 3 sau < -3)')

        # Preg un set pt simulari
        df_outlier_test = df.copy()
        df_numeric = df_outlier_test.select_dtypes(include=[np.number])

        # Calcul Z-score
        z_scores = np.abs(zscore(df_numeric))
        extreme_mask = (z_scores > 3)
        extreme_rows = extreme_mask.any(axis=1)
        df_extreme = df_outlier_test[extreme_rows]

        st.info(f"Număr de rânduri cu valori extreme: {df_extreme.shape[0]}")

        extreme_df_mask = pd.DataFrame(
            extreme_mask,
            index=df_outlier_test.index,
            columns=df_numeric.columns
        )

        # outliers
        styled_extreme_init = df_extreme.style.apply(
            lambda row: [
                'background-color: DarkOrange' if col in extreme_df_mask.columns and extreme_df_mask.loc[
                    row.name, col] else ''
                for col in row.index
            ],
            axis=1
        )

        st.dataframe(styled_extreme_init.format(precision=3), use_container_width=True)

        metoda_extreme = st.radio("Simulează metode de tratare:",
                                  ["Doar evidențiere", "Eliminare", "Înlocuire cu mediană"])
        df_outlier_tratat = df_outlier_test.copy()

        if metoda_extreme == "Eliminare":
            df_outlier_tratat = df_outlier_test[~extreme_rows]
            st.success(f"{df_extreme.shape[0]} rânduri eliminate (simulare).")
            st.markdown("##### Rândurile eliminate:")
            st.dataframe(df_extreme)

        elif metoda_extreme == "Înlocuire cu mediană":
            mediane = df_numeric.median()
            for col in df_numeric.columns:
                mask_col = extreme_mask[:, df_numeric.columns.get_loc(col)]
                df_outlier_tratat.loc[mask_col, col] = mediane[col]

            st.success("Valorile extreme au fost înlocuite cu mediană (simulare).")

            extreme_df_mask = pd.DataFrame(
                extreme_mask,
                index=df_outlier_test.index,
                columns=df_numeric.columns
            )

            styled_extreme = df_outlier_tratat.loc[df_extreme.index].style.apply(
                lambda row: [
                    'background-color: DarkOrange' if col in extreme_df_mask.columns and extreme_df_mask.loc[
                        row.name, col] else ''
                    for col in row.index
                ],
                axis=1
            )

            st.markdown("##### Rânduri modificate (doar celulele modificate sunt evidențiate):")
            st.dataframe(styled_extreme.format(precision=3), use_container_width=True)

        else:
            st.warning("Valorile extreme sunt doar afișate, fără modificări.")

        # Aplicarea DECIZIEI analitice reale

        df_final = df.copy()
        df_numeric_final = df_final.select_dtypes(include=[np.number])
        z_scores_final = np.abs(zscore(df_numeric_final))
        extreme_mask_final = (z_scores_final > 3)

        mediane_final = df_numeric_final.median()
        for col in df_numeric_final.columns:
            mask_col = extreme_mask_final[:, df_numeric_final.columns.get_loc(col)]
            df_final.loc[mask_col, col] = mediane_final[col]

        st.markdown("---")
        st.markdown("##  Decizia finală aplicată pe setul real")

        st.success("🔹 A fost aplicată **înlocuirea valorilor extreme cu mediana** pe coloanele numerice.")

        st.markdown("""
        > Această decizie a fost luată deoarece:
        > - Înlocuirea cu mediana păstrează structura datelor fără a pierde observații.
        > - Valorile extreme pot afecta negativ analiza tendințelor și a corelațiilor.
        """)

        st.markdown("###  Tratarea valorilor cu volum zero")

        st.info("""
        În cadrul setului de date, au fost identificate mai multe rânduri în care volumul tranzacționat (`Volume`) este egal cu `0`.

        Chiar dacă din punct de vedere statistic aceste valori nu au fost marcate ca outlieri (prin metode precum Z-score sau IQR),
        astfel de valori nu aduc informații utile pentru analiza evoluției prețului sau a volumului.

         **Decizie aplicată:** am eliminat din set toate rândurile în care `Volume = 0`.
        """)
        st.dataframe(df[df["Volume"] == 0])

    elif sub_section == 'Metode de codificare a datelor':

        st.markdown('---')
        st.markdown('## Codificarea datelor (Encoding)')
        st.markdown('### Împărțim lunile anului în anotimpuri: Iarna, Primăvara, Vara, Toamna')
        df_encoding = df.copy()
        df_encoding["Luna"] = pd.to_datetime(df_encoding["Date"]).dt.month
        df_encoding = df_encoding[df_encoding["Volume"] > 30000]
        def anotimp(luna):
            if luna in [12, 1, 2]:
                return "Iarna"
            elif luna in [3, 4, 5]:
                return "Primavara"
            elif luna in [6, 7, 8]:
                return "Vara"
            else:
                return "Toamna"

        df_encoding["Anotimp"] = df_encoding["Luna"].apply(anotimp)


        st.markdown("### Mostră random de 12 luni diferite (cu anotimpuri asociate):")
        luni_unice = df_encoding["Luna"].unique()
        mostra_random = pd.concat([df_encoding[df_encoding["Luna"] == luna].sample(1) for luna in luni_unice])
        mostra_random = mostra_random.sort_values("Luna").reset_index(drop=True)
        st.dataframe(mostra_random[["Date", "Luna", "Anotimp"]])


        metoda_encoding = st.radio("Simulează metoda de codificare:", ["Label Encoding", "One-Hot Encoding"])

        if metoda_encoding == "Label Encoding":

            df_encoding_label = df_encoding.copy()
            le = LabelEncoder()
            df_encoding_label["Anotimp_Label"] = le.fit_transform(df_encoding_label["Anotimp"])
            st.markdown("### Rezultat - Label Encoding:")

            st.table(
                df_encoding_label[["Anotimp", "Anotimp_Label"]]
                .drop_duplicates()
                .sort_values("Anotimp_Label")
                .reset_index(drop=True)
            )

            st.markdown("### Număr de intrări per anotimp:")

            st.dataframe(
                df_encoding_label["Anotimp"]
                .value_counts()
                .rename_axis("Anotimp")
                .reset_index(name="Număr apariții")

            )


        else:

            df_encoding_ohe = pd.get_dummies(df_encoding.copy(), columns=["Anotimp"])
            st.markdown("### Rezultat - One-Hot Encoding:")


            explicatie_ohe = pd.DataFrame({
                "Iarna": [1, 0, 0, 0],
                "Primavara": [0, 1, 0, 0],
                "Vara": [0, 0, 1, 0],
                "Toamna": [0, 0, 0, 1]

            }, index=["Iarna", "Primavara", "Vara", "Toamna"])

            st.markdown("### Explicație codificare One-Hot (anotimp → vector binar):")
            st.table(explicatie_ohe)
            st.markdown("Fiecare rând are o coloană activă corespunzătoare anotimpului său:")
            cols_ohe = [col for col in df_encoding_ohe.columns if col.startswith("Anotimp_")]
            df_sample = df_encoding_ohe[["Date"] + cols_ohe].sample(10).reset_index(drop=True)
            st.dataframe(df_sample)

        df_encoding_real = df_encoding.copy()

        df_final = pd.get_dummies(df_encoding_real, columns=["Anotimp"])
        st.markdown("---")
        st.markdown("##  Decizia finală aplicată pentru codificare")
        st.success("🔹 A fost aplicat **One-Hot Encoding** pe coloana 'Anotimp' în setul real (`df_final`).")
        st.markdown("""
        > One-Hot Encoding a fost ales deoarece:
        > - Permite filtrări și agregări mai clare pe anotimpuri distincte.
        """)
        st.dataframe(df_final)

    elif sub_section == "Analiza corelațiilor":

        st.markdown("##  Analiza corelațiilor între variabile")
        st.markdown("""
        Explorăm relațiile dintre variabilele numerice din setul de date pe baza coeficientul de corelație Pearson.

        Această analiză ne ajută să înțelegem ce variabile sunt legate între ele.
        """)

        df_corr = df_final.select_dtypes(include=['float64', 'int64'])

        st.markdown("### Tabelul coeficienților Pearson:")
        st.dataframe(df_corr.corr().round(2))

        st.markdown("### Matrice de corelație (Heatmap):")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Matrice de corelație între variabile")
        st.pyplot(fig)

        st.markdown("### Interpretare:")

        st.markdown("""
        - Variabilele **`Open`**, **`High`**, **`Low`**, **`Close`** și **`Adj Close`** prezintă corelații extrem de ridicate între ele.
          - Acest lucru este normal, considerând faptul că toate aceste coloane reflectă prețuri din aceeași sesiune de tranzacționare.
          - Astfel, va fi suficient să păstrăm o singură variabilă reprezentativă (ex: `Close`).
          - Cu toate acestea, nu le eliminăm pe restul.

        - Variabila **`Volume`** prezintă o **corelație negativă moderată** cu celelalte variabile.
          - Aceasta poate indica faptul că în zilele cu volum mai mare, prețurile tind să fie ușor mai scăzute, sau invers.
          - `Volume` rămâne o variabilă utilă pentru analiză.
        """)



    elif sub_section == "Metode de scalare a datelor":

        st.markdown('## Scalarea datelor numerice')

        df_numeric = df_final.select_dtypes(include=[np.number])
        st.markdown("### Alege coloanele pe care vrei să le scalezi:")

        coloane_scalare = st.multiselect(
            "Coloane disponibile:",
            df_numeric.columns.tolist(),
            default=df_numeric.columns.tolist()
        )

        st.markdown("### Alege metoda de scalare:")
        metoda_scalare = st.radio("Metodă:", ["Min-Max", "Standard (Z-score)", "Robust"])

        if metoda_scalare == "Min-Max":
            scaler = MinMaxScaler()
        elif metoda_scalare == "Standard (Z-score)":
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()

        st.markdown("""
        ** Descriere metode de scalare:**
        - **Min-Max Scaling** → aduce valorile în intervalul `[0, 1]`  
        - **Standard Scaling** → transformă valorile ca Z-score (medie = 0, deviație standard = 1)  
        - **Robust Scaling** → folosește mediană și IQR (ideal pentru date cu outlieri)
        """)


        df_scaled_values = scaler.fit_transform(df_numeric[coloane_scalare])
        df_scaled_result = pd.DataFrame(df_scaled_values, columns=[f"{col}_scaled" for col in coloane_scalare])


        df_scalare_viz = pd.concat([
            df_numeric[coloane_scalare].head(10).reset_index(drop=True),
            df_scaled_result.head(10)
        ], axis=1)

        st.markdown("### Comparație între valorile originale și scalate (primele 10 rânduri):")
        st.dataframe(df_scalare_viz.style.format(precision=3))

        # BOX PLOT COMPARATIV

        col_sample = coloane_scalare[:3]  # max 3 coloane pentru claritate
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].boxplot(df_numeric[col_sample].values, tick_labels=col_sample)
        axs[0].set_title("Distribuție originală")
        axs[1].boxplot(df_scaled_result[[f"{col}_scaled" for col in col_sample]].values, tick_labels=col_sample)
        axs[1].set_title("Distribuție scalată")
        st.markdown("### Compararea distribuției (boxplot):")
        st.pyplot(fig)

        st.info("Aceasta este doar o simulare ce nu modifică efectiv setul de date.")

        st.markdown("---")
        st.markdown("## Decizie privind scalarea în pașii următori")
        st.success(
            "🔹 Având în vedere setul nostru de date, am decis ca în cadrul analizelor să folosim **Standard Scaling (Z-score)**.")
        st.markdown("""
        > StandardScaler normalizează valorile astfel încât:
        > - media fiecărei coloane = 0
        > - deviația standard = 1
        """)

        scaler_model = StandardScaler()
        df_scaled_model = df_final.copy()
        scaled_values_model = scaler_model.fit_transform(df_scaled_model.select_dtypes(include=[np.number]))
        df_scaled_model[[f"{col}_scaled" for col in
                         df_scaled_model.select_dtypes(include=[np.number]).columns]] = scaled_values_model



    elif sub_section == "Prelucrari statistice și agregare":

        st.markdown("## Prelucrări statistice, grupare și agregare")

        st.markdown("""
        > Ne-am propus să explorăm tendințele sezoniere.
        """)

        df_stats = df_final.copy()

        grupare = "Luna"
        if "Anotimp" in df_stats.columns:
            grupare = st.selectbox("🔹 Alege coloana pentru grupare:", ["Anotimp", "Luna"])
        else:
            st.info("Gruparea s-a realizat după coloana `Luna`.")

        coloane_num = df_stats.select_dtypes(include=[np.number]).columns.tolist()
        coloane_num.remove("Luna")
        coloane_alease = st.multiselect("🔸 Alege coloanele numerice:", coloane_num, default=coloane_num)
        functii_disponibile = ["mean", "sum", "min", "max", "std"]
        functii_alease = st.multiselect("🔧 Alege funcțiile de agregare:", functii_disponibile, default=["mean"])

        df_agregat = df_stats.groupby(grupare, as_index=False)[coloane_alease].agg(functii_alease)
        st.markdown("###  Tabelul rezultat (cu agregări):")
        st.dataframe(df_agregat.style.format(precision=2).background_gradient(cmap="Blues", axis=None))

        if len(coloane_alease) > 0:
            col_grafic = st.selectbox(" Alege o coloană pentru grafic:", coloane_alease)
            if len(functii_alease) > 1:
                functie_grafic = st.selectbox(" Alege funcția pentru grafic:", functii_alease)
            else:
                functie_grafic = functii_alease[0]
            tip_grafic = st.radio(" Tip grafic:", ["Bar Chart", "Line Chart"], horizontal=True)
            x_vals = df_agregat[grupare]
            try:
                if isinstance(df_agregat.columns, pd.MultiIndex):
                    y_vals = df_agregat[(col_grafic, functie_grafic)]
                    titlu = f"{functie_grafic.capitalize()} pentru '{col_grafic}' după {grupare}"

                else:
                    y_vals = df_agregat[col_grafic]
                    titlu = f"{functie_grafic.capitalize()} pentru '{col_grafic}' după {grupare}"

                fig, ax = plt.subplots()
                ax.set_title(titlu)
                ax.set_xlabel(grupare)
                ax.set_ylabel(f"{col_grafic} ({functie_grafic})")
                if tip_grafic == "Bar Chart":
                    ax.bar(x_vals, y_vals)

                else:
                    ax.plot(x_vals, y_vals, marker='o')

                st.pyplot(fig)


            except Exception as e:
                st.warning(f"Nu s-a putut genera graficul: {e}")

        if len(coloane_alease) == 1:
            col_box = coloane_alease[0]

            st.markdown("### Boxplot pe grupuri")
            fig2, ax2 = plt.subplots()
            df_boxplot = df_stats[[grupare, col_box]]
            df_boxplot.boxplot(by=grupare, column=col_box, ax=ax2)
            ax2.set_title(f"Distribuția '{col_box}' în funcție de {grupare}")
            ax2.set_ylabel(col_box)
            ax2.set_xlabel(grupare)
            st.pyplot(fig2)



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