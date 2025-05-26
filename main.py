import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import osmnx as ox
import os
from pathlib import Path
import folium
from folium.features import GeoJsonTooltip,GeoJson
from streamlit_folium import st_folium
from shapely.geometry import Point
import contextily
from shapely.ops import nearest_points
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, \
    f1_score


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



# Încărcare hartă mondială
def load_or_save_gis(name: str, url: str, folder="data/maps") -> gpd.GeoDataFrame:
    os.makedirs(folder, exist_ok=True)
    shp_path = Path(folder) / f"{name}.shp"
    if shp_path.exists():
        return gpd.read_file(shp_path)

    gdf = gpd.read_file(url)
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    return gdf

def load_or_save_rivers(name: str,folder="data/maps"):
    os.makedirs(folder, exist_ok=True)
    shp_path = Path(folder) / f"{name}_rivers.shp"
    if shp_path.exists():
        return gpd.read_file(shp_path)
    r = ox.features_from_place(f"{name}", {"waterway": "river"})
    r.to_file(shp_path, driver="ESRI Shapefile")
    return r

#Definirea hartilor:
world = load_or_save_gis("world_boundary",

                                 "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")

cities = load_or_save_gis("cities_detailed",
                                  "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip")


# Citire și conversie corectă pentru compatibilitate completă
df = pd.read_csv('data/Samsung.csv')
df_phones = pd.read_csv('data/PhoneBrandsWorld.csv', sep=';')
df_phones.columns = df_phones.columns.str.strip()

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

    sub_section = st.sidebar.radio("Secțiuni din proiect", ["Prezentare date", "Filtrări pe baza datelor", "Tratarea valorilor lipsă și a valorilor extreme", "Metode de codificare a datelor", "Analiza corelațiilor","Metode de scalare a datelor", "Prelucrari statistice și agregare"
                                                            ,"Harta interactiva a distributiei globale a brandurilor de telefoane", "Harta râurilor din Coreea de Sud",
                                                            "Vecinii Coreei de Sud", "Analiza oraselor Coreei de Sud", "GDP Asia",
                                                            "Clusterizare KMeans","Regresie logistică","Regresie multiplă"])

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

        coloane_disponibile = [col for col in df_numeric.columns if col != "Luna"]

        coloane_scalare = st.multiselect(
            "Coloane disponibile:",
            coloane_disponibile,
            default=coloane_disponibile
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




    elif sub_section == "Harta interactiva a distributiei globale a brandurilor de telefoane":

        st.markdown("## Branduri dominante de telefoane pe glob")

        tema_harta = st.selectbox("Alege stilul hărții", ["Light", "Dark"], index=0)

        # Conversie text la float și apoi la text cu %
        for col in ['#1 Market Share', '#2 Market Share', '#3 Market Share']:
            df_phones[col] = df_phones[col].str.replace('%', '').astype(float)
            df_phones[f"{col} (%)"] = df_phones[col].apply(lambda x: f"{x:.2f}%")

        world = load_or_save_gis("world_boundary",
                                 "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")

        df_phones.columns = df_phones.columns.str.strip().str.replace('\ufeff', '')
        df_merged = world.merge(df_phones, left_on="ADMIN", right_on="Country", how="inner")

        # Mapare culori pentru branduri
        color_map = {
            "Apple": "#5C6BC0", "Samsung": "#26A69A", "Xiaomi": "#FFD700", "Google": "#8BC34A",
            "Huawei": "#FF8A65", "Motorola": "#90A4AE", "Oppo": "#EC407A", "Vivo": "#4DD0E1",
            "Tecno": "#7E57C2", "Infinix": "#A1887F", "LG": "#E57373", "itel": "#BCAAA4", "Unknown": "#FFA726"
        }

        df_merged["color"] = df_merged["#1 Brand"].map(color_map).fillna("#E0E0E0")

        # Alegere stil hartă și contrast
        tile_style = "cartodbpositron" if tema_harta == "Light" else "cartodbdark_matter"
        tooltip_bg = "white" if tema_harta == "Light" else "#1c1c1c"
        tooltip_text = "#222" if tema_harta == "Light" else "#EEE"
        tooltip_shadow = "rgba(0,0,0,0.2)" if tema_harta == "Light" else "rgba(255,255,255,0.1)"

        # Legendă afișată deasupra hărții
        st.markdown("### Legendă branduri dominante")
        cols = st.columns(4)  # Împarte pe 4 coloane pentru afișare compactă

        for i, (brand, color) in enumerate(color_map.items()):
            with cols[i % 4]:
                st.markdown(f"<div style='display: flex; align-items: center;'>"
                            f"<div style='width: 14px; height: 14px; background-color: {color}; "
                            f"margin-right: 6px; border: 1px solid #aaa; display: inline-block;'></div>"
                            f"<span style='font-size: 13px;'>{brand}</span></div>",
                            unsafe_allow_html=True)

        # Inițializare hartă
        m = folium.Map(location=[15, 0], zoom_start=2, tiles=tile_style)

        folium.GeoJson(
            data=df_merged,
            name="Țări",
            style_function=lambda x: {
                "fillColor": x["properties"]["color"],
                "color": "#444",
                "weight": 1.2,
                "fillOpacity": 0.9,
            },
            highlight_function=lambda x: {
                "color": "black",
                "weight": 2.5,
                "fillOpacity": 0.95,
            },
            tooltip=GeoJsonTooltip(
                fields=["ADMIN", "#1 Brand", "#1 Market Share (%)", "#2 Brand", "#2 Market Share (%)", "#3 Brand",
                        "#3 Market Share (%)"],
                aliases=["Țară", "Brand #1", "Cota #1", "Brand #2", "Cota #2", "Brand #3", "Cota #3"],
                sticky=True,
                style=(
                    f"background-color: {tooltip_bg}; "
                    f"color: {tooltip_text}; "
                    f"font-family: Arial; "
                    f"font-size: 13px; "
                    f"padding: 10px; "
                    f"box-shadow: 2px 2px 6px {tooltip_shadow};"
                )
            ),
            control=False,
            overlay=True,
            show=True,
            interactive=True
        ).add_to(m)

        # Afișare hartă
        st_folium(m, use_container_width=True, height=800)

    elif sub_section == "Harta râurilor din Coreea de Sud":
        # 1. Descarcă contururile țărilor de pe NACIS


        cities = cities[cities["SOV0NAME"] == "Korea, South"]

        south_korea = world[world["ADMIN"] == "South Korea"]
        north_korea = world[world["ADMIN"] == "North Korea"]

        # Găsește vecinii Coreei de Sud
        vecini = world[world.touches(south_korea.iloc[0].geometry)]

        # Combină doar aceste geometrii
        combinat = gpd.GeoDataFrame(pd.concat([vecini, south_korea]), crs=world.crs)

        neighbours = vecini.to_crs(epsg=32635)  # pentru afișare pe hartă

        # 2. Descarcă râurile (lake centerlines)
        rivers =load_or_save_rivers("South Korea")

        # 3. Asigură-te că sunt în același sistem de coordonate
        rivers = rivers.to_crs(south_korea.crs)

        # 4. Intersecția râurilor cu Coreea de Sud
        rivers_in_korea = gpd.overlay(rivers, south_korea, how="intersection")

        # 5. Afișează harta
        fig, ax = plt.subplots(figsize=(10, 10))
        south_korea.plot(ax=ax, color="lightgrey")
        rivers_in_korea.plot(ax=ax, color="blue", linewidth=1, label="Rivers")
        cities.plot(ax=ax, color="red", markersize=20, label="Cities")

        ax.set_title("Râurile din Coreea de Sud", fontsize=14)
        plt.axis("off")
        st.pyplot(fig)

        # 1. Ne asigurăm că atât râurile cât și vecinii sunt în același CRS (EPSG:32635)
        rivers = rivers[['geometry']].copy()
        rivers = rivers.to_crs(epsg=32635)
        neighbours = neighbours.to_crs(epsg=32635)

        # 2. Intersecția râurilor cu vecinii Coreei de Sud
        rivers_crossing = gpd.sjoin(neighbours, rivers, how='inner', predicate='intersects')

        # 3. Grupare: câte râuri intersectează fiecare țară vecină
        rauri_pe_tari = rivers_crossing.groupby('ADMIN').size().reset_index(name='Număr râuri')
        rauri_pe_tari.rename(columns={'ADMIN': 'Vecini'}, inplace=True)

        # 4. Afișare în Streamlit
        st.markdown("### Numărul de râuri care traversează fiecare vecin al Coreei de Sud")
        st.dataframe(rauri_pe_tari, hide_index=True)


        ####RAURILE CARE TRAVERSEAZA TARILE
        rivers = rivers.to_crs(epsg=32635)
        neighbours = neighbours.to_crs(epsg=32635)
        rivers_crossing = gpd.sjoin(neighbours, rivers, how='inner', predicate='intersects')

        # 1. Selectăm doar Coreea de Sud și Coreea de Nord din world
        sk = world[world["ADMIN"] == "South Korea"]
        nk = world[world["ADMIN"] == "North Korea"]
        coreea = gpd.GeoDataFrame(pd.concat([sk, nk]), crs=world.crs)

        # 2. Reproiectăm toate la EPSG:32652 (UTM Coreea)
        coreea = coreea.to_crs(epsg=32652)
        rivers = rivers.to_crs(epsg=32652)

        # 3. Păstrăm doar râurile care intersectează Coreea de Sud sau Coreea de Nord
        rivers_in_coreea = gpd.sjoin(rivers, coreea, how='inner', predicate='intersects')

        # 4. Plot
        fig, ax = plt.subplots(figsize=(10, 14))  # harta mai mare

        # Harta Coreei
        coreea.plot(ax=ax, color='lightgray', edgecolor='black', label='Coreea de Sud & Nord')

        # Contur Coreea de Sud și Nord
        sk.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, label='Coreea de Sud')
        nk.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2, label='Coreea de Nord')

        # Râurile
        rivers_in_coreea.plot(ax=ax, color='darkblue', linewidth=1, label='Râuri')

        # Fixare axă doar pe zona relevantă
        ax.set_xlim(coreea.total_bounds[0] - 10000, coreea.total_bounds[2] + 10000)
        ax.set_ylim(coreea.total_bounds[1] - 10000, coreea.total_bounds[3] + 10000)

        # Final
        plt.title("Râurile din Coreea de Sud și Coreea de Nord")
        plt.axis('off')
        st.pyplot(fig)

    elif sub_section == "Vecinii Coreei de Sud":

        # Încarcă și reproiectează vecinii




        # Încarcă și Coreea de Sud


        south_korea = world[world["ADMIN"] == "South Korea"]

        # Găsește vecinii Coreei de Sud
        vecini = world[world.touches(south_korea.iloc[0].geometry)]

        # Combină doar aceste geometrii
        combinat = gpd.GeoDataFrame(pd.concat([vecini, south_korea]), crs=world.crs)

        neighbours = vecini.to_crs(epsg=32635)  # pentru afișare pe hartă

        neighbours["area_km2"] = neighbours.to_crs(epsg=32652).geometry.area / 1_000_000

        # Use a more appropriate CRS for South Korea (UTM zone 52N)

        south_korea_proj = south_korea.to_crs(epsg=32652)  # UTM zone 52N covers most of South Korea

        south_korea_area = south_korea_proj.geometry.area / 1_000_000

        # For display on the map

        south_korea = south_korea.to_crs(epsg=32635)

        south_korea["area_km2"] = south_korea_area

        # Hartă folium

        m = folium.Map(location=[36.5, 127.5], zoom_start=6)

        # Colorează vecinii

        folium.Choropleth(

            geo_data=neighbours,
            data=neighbours,
            columns=["ADMIN", "area_km2"],
            key_on="feature.properties.ADMIN",
            legend=False,
            fill_color="YlGnBu",
            line_opacity=0.2,
            fill_opacity=0.7,
            name="Vecini"

        ).add_to(m)

        # Tooltip pe vecini

        GeoJson(
            neighbours,
            tooltip=GeoJsonTooltip(fields=["ADMIN", "area_km2"],

                                   aliases=["Țară", "Suprafață (km²)"],

                                   localize=True)

        ).add_to(m)

        # Colorează Coreea de Sud

        GeoJson(

            south_korea,

            style_function=lambda x: {"fillColor": "orange", "color": "black", "weight": 2, "fillOpacity": 0.6},

            tooltip=GeoJsonTooltip(fields=["ADMIN", "area_km2"],

                                   aliases=["Coreea de Sud", "Suprafață (km²)"],

                                   localize=True)

        ).add_to(m)

        # Titlu și afișare

        st.title("Vecinii Coreei de Sud și Granița cu Coreea de Nord")

        st_folium(m, use_container_width=True, height=600)

        st.markdown("### Calcularea centrilor din Coreea de Sud cu Coreea de Nord")

        south_korea = world[world["ADMIN"] == "South Korea"]



        # Reproiectează DOAR zona de interes într-un CRS metric
        combinat_metric = combinat.to_crs(epsg=32652)

        # Calculează centroizii în sistem proiectat
        combinat_metric["centroid"] = combinat_metric.geometry.centroid
        centroizi = combinat_metric.set_geometry("centroid")
        # Reproiectăm înapoi în EPSG:4326 doar pentru afișare, dacă vrei coordonate geografice
        combinat = combinat_metric.to_crs(epsg=4326)
        centroizi = centroizi.to_crs(epsg=4326)

        # Plot în matplotlib
        fig, ax = plt.subplots()
        combinat.plot(ax=ax, edgecolor="black", facecolor="none")
        centroizi.plot(ax=ax, color="red", markersize=10)
        st.pyplot(fig)

        # Coordonatele Seulului
        seoul_wgs = Point(126.9780, 37.5665)

        # Convertim în EPSG:32652 (Coreea)
        seoul_utm = gpd.GeoSeries([seoul_wgs], crs="EPSG:4326").to_crs(epsg=32652).iloc[0]

        # Selectăm Coreea de Nord și centroidul ei
        north_korea = world[world["ADMIN"] == "North Korea"].to_crs(epsg=32652)
        nk_centroid = north_korea.geometry.centroid.iloc[0]

        # Calculăm distanța
        dist_km = seoul_utm.distance(nk_centroid) / 1000

        # Creăm tabelul
        df_dist = pd.DataFrame([{
            "Oraș": "Seul",
            "Țintă": "Centru Coreea de Nord",
            "Distanță (km)": round(dist_km, 2)
        }])

        # Afișăm tabelul
        st.markdown("### Distanța Seul – Coreea de Nord")
        st.dataframe(df_dist,hide_index=True)

        # Creează figura și axa
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotează cu coloană "name" (sau "ADMIN", în funcție de ce ai)
        combinat.plot(column='ADMIN', edgecolor='black', legend=True, ax=ax)

        # Afișează în Streamlit
        st.pyplot(fig)

        #Afisare vecini
        df_vecini = pd.DataFrame(vecini["ADMIN"].values, columns=["Vecini"])

        # Afișăm fără index
        st.markdown("### Țările vecine Coreei de Sud")
        st.dataframe(df_vecini, hide_index=True)

        # 1. Creează punctul Seul în WGS84
        seoul = gpd.GeoDataFrame(
            {'name': ['Seoul']},
            geometry=[Point(126.9780, 37.5665)],
            crs="EPSG:4326"
        )

        # 2. Selectează doar Coreea de Sud și Coreea de Nord
        sk = world[world["ADMIN"] == "South Korea"]
        nk = world[world["ADMIN"] == "North Korea"]
        coreea = pd.concat([sk, nk])
        coreea = gpd.GeoDataFrame(coreea, crs=world.crs)

        # 3. Conversie la EPSG:3857 pentru contextily
        seoul = seoul.to_crs(epsg=3857)
        coreea_ctx = coreea.to_crs(epsg=3857)

        # 4. Calculează centroizii
        coreea_ctx["centroid"] = coreea_ctx.geometry.centroid

        # 5. Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Colorează Coreea de Sud și Nord diferit
        coreea_ctx.plot(column='ADMIN', ax=ax, alpha=0.5, edgecolor='black', legend=True)

        # Marchează centroizii cu galben
        coreea_ctx["centroid"].plot(ax=ax, color="yellow")

        # Marchează Seulul
        seoul.plot(ax=ax, color='black', marker='*', markersize=100)

        # Fundal hărți reale
        contextily.add_basemap(ax, crs=coreea_ctx.crs.to_string())

        # Titlu și afișare
        plt.title('Coreea de Sud, Coreea de Nord și Seul')
        plt.axis('off')
        st.pyplot(fig)
    elif sub_section=="Analiza oraselor Coreei de Sud":

        # 1. Selectăm doar Coreea de Sud
        south_korea = world[world["ADMIN"] == "South Korea"]

        # 2. Selectăm orașele doar din Coreea de Sud
        cities_sk = cities[cities["SOV0NAME"] == "Korea, South"]

        # 3. Convertim în EPSG:32652 (metric, pentru Coreea)
        sk_utm = south_korea.to_crs(epsg=32652)
        cities_utm = cities_sk.to_crs(epsg=32652)

        # 4. Creăm buffer de 3 km în jurul fiecărui oraș
        buffers = cities_utm.copy()
        buffers["geometry"] = cities_utm.buffer(3000)

        # 5. Calculăm zonele din Coreea de Sud care NU sunt în aceste buffers
        fabrici_potentiale = gpd.overlay(sk_utm, buffers, how='difference')

        # 6. Plot în matplotlib
        fig, ax = plt.subplots(figsize=(10, 10))
        fabrici_potentiale.plot(ax=ax, alpha=0.6, edgecolor='black', facecolor='lightgreen')
        cities_utm.plot(ax=ax, color='red', markersize=10, label='Fabrici')
        sk_utm.boundary.plot(ax=ax, edgecolor='black')

        # Titlu și afișare
        ax.set_title("Zone din Coreea de Sud pentru potențiale fabrici (minim 3 km distanță de orașe)")
        plt.axis('off')
        plt.legend()
        st.pyplot(fig)

        ###Distanta orase Coreea de Sud vs Corea de Nord
        # 1. Selectăm doar orașele din Coreea de Sud
        cities_sk = cities[cities["SOV0NAME"] == "Korea, South"].to_crs(epsg=4326)

        # 2. Selectăm doar Coreea de Nord
        north_korea = world[world["ADMIN"] == "North Korea"].to_crs(epsg=4326)

        # 3. Obținem geometria combinată a Coreei de Nord
        nk_geometry = north_korea.union_all()

        # 4. Calculează distanța de la fiecare oraș la Coreea de Nord
        results = []

        for idx, city in cities_sk.iterrows():
            city_point = city.geometry
            coord_city = (city_point.y, city_point.x)

            # Cel mai apropiat punct de pe granița nord-coreeană
            _, p_nk = nearest_points(city_point, nk_geometry)
            coord_nk = (p_nk.y, p_nk.x)

            distance_km = geodesic(coord_city, coord_nk).km

            results.append({
                "Oraș": city["NAME"],
                "Distanță până la Coreea de Nord (km)": round(distance_km, 2)
            })

        # 5. Convertim în DataFrame și sortăm
        df_distante = pd.DataFrame(results).sort_values(by="Distanță până la Coreea de Nord (km)")

        # 6. Afișăm în Streamlit
        st.markdown("### Distanța de la fiecare oraș din Coreea de Sud până la granița cu Coreea de Nord")
        st.dataframe(df_distante, hide_index=True)

    elif sub_section=="GDP Asia":
        asia = world[world["CONTINENT"] == "Asia"].copy()

        # 3. Încarcă fișierul cu GDP numeric
        gdp = pd.read_csv("data/asian_gdp_clean.csv")  # asigură-te că se potrivește coloana "Country"

        # 4. Merge între datele geografice și cele economice
        merged = asia.merge(gdp, left_on="ADMIN", right_on="Country", how="left")

        proj = merged.to_crs(epsg=3035)
        centroids = proj.geometry.centroid.to_crs(merged.crs)

        # Plot
        fig, ax = plt.subplots(figsize=(15, 10))
        merged.plot(color='lightgray', edgecolor='black', ax=ax)
        merged[merged['GDP_USD'].notna()].plot(column='GDP_USD', cmap='viridis', legend=True,
                                               ax=ax, edgecolor='black',
                                               legend_kwds={'label': "GDP (USD)", 'orientation': "horizontal"})

        for x, y, label in zip(centroids.x, centroids.y, proj["ADMIN"]):
            ax.text(x, y, label, fontsize=7, ha='center', va='center')

        ax.set_title("GDP of Asian Countries")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()

        # Asta e linia corectă în Streamlit
        st.pyplot(fig)

    elif sub_section == "Clusterizare KMeans":

        st.markdown("## Analiză de tip Clusterizare (KMeans)")

        st.markdown("""

                În această secțiune aplicăm algoritmul **KMeans** pentru a grupa zilele de tranzacționare în funcție de două variabile relevante:

                - `Close` (prețul de închidere)

                - `Volume` (volumul tranzacționat)


                Vom folosi setul de date scalat în secțiunile anterioare.

                """)

        # selectare variabile

        try:

            X = df_scaled_model[["Close", "Volume"]].values

        except KeyError:

            st.error("Coloanele 'Close' și 'Volume' nu sunt prezente în setul scalat.")

            st.stop()

        # elbow method - calcul wcss

        wcss = []

        for k in range(1, 11):
            model = KMeans(n_clusters=k, init='k-means++', random_state=42)

            model.fit(X)

            wcss.append(model.inertia_)

        st.markdown("### Elbow Method – alegerea vizuală a lui k:")

        fig_elbow, ax_elbow = plt.subplots()

        sns.lineplot(x=range(1, 11), y=wcss, marker='o', ax=ax_elbow, color='crimson')

        ax_elbow.set_title("Elbow Method – WCSS în funcție de k")

        ax_elbow.set_xlabel("Numărul de clustere (k)")

        ax_elbow.set_ylabel("WCSS (Within-Cluster Sum of Squares)")

        st.pyplot(fig_elbow)

        # silhouette Score pentru k de la 2 la 10

        silhouette_scores = {}

        for k in range(2, 11):
            model = KMeans(n_clusters=k, init='k-means++', random_state=42)

            labels = model.fit_predict(X)

            score = silhouette_score(X, labels)

            silhouette_scores[k] = score

        k_optimal = max(silhouette_scores, key=silhouette_scores.get)

        st.markdown("### Silhouette Scores pentru k = 2..10:")

        """
        Măsoară cât de apropiat este un punct de clusterul său comparativ cu celelalte clustere.
        Scorul este între -1 și 1:

        ~1 → punctul este bine încadrat

        ~0 → este la graniță între clustere

        < 0 → probabil este pus greșit în cluster"""

        st.table(pd.DataFrame.from_dict(silhouette_scores, orient='index', columns=["Silhouette Score"]).round(4))

        st.success(f" Numărul optim de clustere, conform scorului Silhouette, este: **k = {k_optimal}**")

        # antrenare finala

        model_final = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42)

        cluster_labels = model_final.fit_predict(X)

        silhouette_final = silhouette_score(X, cluster_labels)

        st.markdown(f"### Silhouette Score final: `{silhouette_final:.4f}`")

        # adaugare in df final

        df_final["Cluster"] = cluster_labels

        # scatterplot

        st.markdown("### Vizualizarea clusterelor folosind scatterplot: ")

        fig_clust, ax = plt.subplots(figsize=(10, 6))

        palette = sns.color_palette("Set2", k_optimal)

        for label in range(k_optimal):
            sns.scatterplot(

                x=X[cluster_labels == label, 0],

                y=X[cluster_labels == label, 1],

                s=50,

                label=f"Cluster {label + 1}",

                ax=ax,

                color=palette[label]

            )

        # centroide

        sns.scatterplot(

            x=model_final.cluster_centers_[:, 0],

            y=model_final.cluster_centers_[:, 1],

            s=200,

            color='red',

            label='Centroide',

            marker='X',

            ax=ax

        )

        ax.set_xlabel("Close (scalat)")

        ax.set_ylabel("Volume (scalat)")

        ax.set_title("Reprezentarea vizuală a clusterelor KMeans")

        ax.grid(True)

        st.pyplot(fig_clust)

        st.markdown("## Interpretarea clusterelor identificate")

        if "Cluster" not in df_final.columns:
            st.warning("Eticheta de cluster nu este disponibilă. Te rugăm să efectuezi întâi clusterizarea.")
            st.stop()

        st.markdown("### Număr de observații per cluster")
        st.dataframe(
            df_final["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Număr de rânduri"))

        st.markdown("### Statistici descriptive pe fiecare cluster")

        media_pe_cluster = df_final.groupby("Cluster").mean(numeric_only=True).round(2)
        st.dataframe(media_pe_cluster)

        st.markdown("""
                    > Valorile afișate mai sus reprezintă mediile variabilelor numerice din fiecare cluster.  
                    Acestea ne pot ajuta să interpretăm semnificația fiecărui grup:
                    - Clustere cu `Close` mare și `Volume` mic → zile scumpe cu activitate redusă
                    - Clustere cu `Volume` mare → zile foarte active, posibil în perioade volatile
                    """)

        st.markdown("### Vizualizarea primelor 3 rânduri din fiecare cluster")
        cluster_ids = sorted(df_final["Cluster"].unique())

        for cid in cluster_ids:
            st.markdown(f"#### Cluster {cid}")
            exemple = df_final[df_final["Cluster"] == cid].head(3).reset_index(drop=True)
            st.dataframe(exemple)

        st.markdown("""

                >  Clusterizarea ne ajută să identificăm tipare în comportamentul zilnic de tranzacționare, 

                cum ar fi grupuri cu volum ridicat și preț redus, sau invers.  

                Eticheta de cluster a fost salvată în setul final (`df_final["Cluster"]`).

                """)




    elif sub_section == "Regresie logistică":

        st.markdown("## Regresie Logistică – Predicția trendului pozitiv în următoarele 5 zile")

        st.markdown("""

                Reformulăm problema: prezicem dacă trendul general în următoarele **5 zile** va fi pozitiv.


                - `TrendPozitiv_5zile = 1` dacă media `Close` din următoarele 5 zile este mai mare decât prețul de azi

                - `TrendPozitiv_5zile = 0` altfel


                """)

        df_trend5 = df_final.copy()

        df_trend5["Return"] = df_trend5["Close"].pct_change()

        df_trend5["MA_3"] = df_trend5["Close"].rolling(3).mean()

        df_trend5["MA_3_diff"] = df_trend5["Close"] - df_trend5["MA_3"]

        df_trend5["Volume_change"] = df_trend5["Volume"].pct_change()

        df_trend5["TrendPozitiv_5zile"] = (

                (df_trend5["Close"].shift(-1) + df_trend5["Close"].shift(-2) +

                 df_trend5["Close"].shift(-3) + df_trend5["Close"].shift(-4) +

                 df_trend5["Close"].shift(-5)) / 5 > df_trend5["Close"]

        ).astype(int)

        df_trend5.dropna(inplace=True)

        st.markdown("### Distribuția clasei țintă `TrendPozitiv_5zile`:")

        distributie = df_trend5["TrendPozitiv_5zile"].value_counts(normalize=True).rename("Proporție (%)") * 100

        st.dataframe(distributie)

        feature_cols = ["Return", "MA_3_diff", "Volume_change", "Luna",

                        "Anotimp_Iarna", "Anotimp_Primavara", "Anotimp_Vara", "Anotimp_Toamna"]

        X = df_trend5[feature_cols]

        y = df_trend5["TrendPozitiv_5zile"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = LogisticRegression(class_weight='balanced', max_iter=1000)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        st.markdown("### Rezultatele modelului:")

        st.write("**Acuratețe:**", f"{accuracy_score(y_test, y_pred):.4f}")

        st.write("**F1 Score:**", f"{f1_score(y_test, y_pred):.4f}")

        st.write("**ROC AUC Score:**", f"{roc_auc_score(y_test, y_prob):.4f}")

        st.markdown("### Matricea de confuzie:")

        cm = confusion_matrix(y_test, y_pred)

        fig_cm, ax_cm = plt.subplots()

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)

        ax_cm.set_xlabel("Valori prezise")

        ax_cm.set_ylabel("Valori reale")

        st.pyplot(fig_cm)

        st.markdown("### Curba ROC:")

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        fig_roc, ax_roc = plt.subplots()

        ax_roc.plot(fpr, tpr, label="Logistic Regression", color='blue')

        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random")

        ax_roc.set_title("Curba ROC – Predicție trend 5 zile")

        ax_roc.set_xlabel("False Positive Rate")

        ax_roc.set_ylabel("True Positive Rate")

        ax_roc.legend()

        st.pyplot(fig_roc)

        st.markdown("### Interpretare:")

        st.success("""

                Modelul logistic aplicat pe `TrendPozitiv_5zile` oferă o performanță modestă, cu un AUC ușor peste 0.5 și o matrice de confuzie echilibrată (~270 predicții corecte pentru fiecare clasă).

                """)



    elif sub_section == "Regresie multiplă":

        st.markdown("## Regresie Liniară Multiplă")

        st.markdown("""

                Am construit un model de regresie liniară multiplă având ca variabilă dependentă `Close`, iar ca variabile explicative:

                - `Low` – prețul minim al zilei

                - `Volume` – volumul tranzacționat

                - `Luna` – luna calendaristică (pentru surprinderea sezonalității)


                """)

        X = df_final[["Low", "Volume", "Luna"]]

        y = df_final["Close"]

        X_const = sm.add_constant(X)

        model = sm.OLS(y, X_const).fit()

        y_pred = model.predict(X_const)

        st.markdown("### Rezumatul modelului:")

        st.text(model.summary())

        rmse = np.sqrt(mean_squared_error(y, y_pred))

        st.markdown(f"**RMSE (Root Mean Squared Error):** `{rmse:.2f}`")

        st.markdown("### Grafic: Valori reale vs. Valori prezise")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.scatter(y, y_pred, alpha=0.5)

        ax.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", linewidth=2)

        ax.set_xlabel("Valori reale (Close)")

        ax.set_ylabel("Valori prezise (Close)")

        ax.set_title("Regresie multiplă: Valori reale vs. prezise")

        st.pyplot(fig)

        st.markdown("### Interpretarea modelului")

        st.success("""
                Modelul de regresie obținut explică foarte bine variația prețului de închidere (`Close`), având un coeficient de determinare **R² ≈ 0.994**. 
                Aceasta înseamnă că aproximativ 99,4% din variația prețului `Close` este explicată prin combinația liniară a celor trei predictori aleși:

                - **`Low`** – are o relație aproape perfectă cu `Close`, ceea ce era de așteptat deoarece în piețele bursiere prețul de închidere este, de obicei, apropiat de minimul zilei.
                - **`Volume`** – are o influență negativă semnificativă, indicând că un volum mare poate semnala vânzări în exces sau corecții.
                - **`Luna`** – aduce o contribuție pozitivă moderată, surprinzând o posibilă sezonalitate în evoluția prețurilor.

                Cu toate acestea, modelul trebuie interpretat cu precauție din cauza posibilei **multicoliniarități** între variabilele de tip preț (`Open`, `High`, `Low`, `Close`), ceea ce este o realitate structurală în datele financiare.
                """)








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