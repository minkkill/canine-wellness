import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🐕 Дашборд анализа здоровья собак",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #A23B72;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #F18F01;
        padding-bottom: 0.5rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .unhealthy-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🐕 Дашборд анализа здоровья собак</h1>', unsafe_allow_html=True)

breed_translation = {
    'Labrador Retriever': 'Лабрадор-ретривер',
    'German Shepherd': 'Немецкая овчарка',
    'Golden Retriever': 'Золотистый ретривер',
    'Bulldog': 'Бульдог',
    'Beagle': 'Бигль',
    'Poodle': 'Пудель',
    'Rottweiler': 'Ротвейлер',
    'Yorkshire Terrier': 'Йоркширский терьер',
    'Boxer': 'Боксер',
    'Dachshund': 'Такса',
}

sex_translation = {
    'Male': 'Самец',
    'Female': 'Самка',
    'Unknown': 'Неизвестно'
}

size_translation = {
    'Small': 'Маленький',
    'Medium': 'Средний',
    'Large': 'Большой',
    'Unknown': 'Неизвестно'
}

activity_translation = {
    'Low': 'Низкий',
    'Moderate': 'Умеренный',
    'High': 'Высокий',
    'Unknown': 'Неизвестно'
}

diet_translation = {
    'Dry Food': 'Сухой корм',
    'Wet Food': 'Влажный корм',
    'Raw': 'Сырое питание',
    'Unknown': 'Неизвестно'
}

spay_translation = {
    'Spayed/Neutered': 'Кастрирован',
    'Intact': 'Не кастрирован',
    'Unknown': 'Неизвестно'
}

@st.cache_data
def load_data():
    df = pd.read_csv('data/synthetic_dog_breed_health_data.csv')
    df['Weight (kg)'] = df['Weight (lbs)'] * 0.453592
    df = df.rename(columns={
        'Breed': 'Порода',
        'Breed Size': 'Размер породы',
        'Sex': 'Пол',
        'Age': 'Возраст',
        'Weight (kg)': 'Вес (кг)',
        'Daily Activity Level': 'Уровень активности',
        'Diet': 'Диета',
        'Spay/Neuter Status': 'Статус кастрации',
        'Annual Vet Visits': 'Посещений ветеринара в год',
        'Healthy': 'Здоров'
    })
    df['Порода'] = df['Порода'].map(breed_translation).fillna('Неизвестно')
    df['Пол'] = df['Пол'].map(sex_translation).fillna('Неизвестно')
    df['Размер породы'] = df['Размер породы'].map(size_translation).fillna('Неизвестно')
    df['Уровень активности'] = df['Уровень активности'].map(activity_translation).fillna('Неизвестно')
    df['Диета'] = df['Диета'].map(diet_translation).fillna('Неизвестно')
    df['Статус кастрации'] = df['Статус кастрации'].map(spay_translation).fillna('Неизвестно')
    df['Здоров'] = df['Здоров'].map({'Yes': 'Да', 'No': 'Нет'})
    df['Порода'] = df['Порода'].fillna('Неизвестно')
    df['Размер породы'] = df['Размер породы'].fillna('Неизвестно')
    df['Пол'] = df['Пол'].fillna('Неизвестно')
    df['Уровень активности'] = df['Уровень активности'].fillna('Неизвестно')
    df['Диета'] = df['Диета'].fillna('Неизвестно')
    df['Статус кастрации'] = df['Статус кастрации'].fillna('Неизвестно')
    return df

df = load_data()

st.sidebar.markdown("## 🎛️ Панель управления")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🔍 Фильтры данных")
breeds = st.sidebar.multiselect(
    "Выберите породы:",
    options=df['Порода'].unique(),
    default=df['Порода'].unique(),
    help="Выберите породы собак для анализа"
)

age_range = st.sidebar.slider(
    "Возраст (лет):",
    min_value=int(df['Возраст'].min()),
    max_value=int(df['Возраст'].max()),
    value=(int(df['Возраст'].min()), int(df['Возраст'].max())),
    help="Выберите диапазон возраста"
)

filtered_df = df[
    (df['Порода'].isin(breeds)) &
    (df['Возраст'] >= age_range[0]) &
    (df['Возраст'] <= age_range[1])
].copy()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-container">
        <h3>📊 Всего собак</h3>
        <h2>{}</h2>
    </div>
    """.format(len(filtered_df)), unsafe_allow_html=True)

with col2:
    healthy_pct = (filtered_df['Здоров'] == 'Да').sum() / len(filtered_df.dropna(subset=['Здоров'])) * 100
    st.markdown("""
    <div class="metric-container">
        <h3>💚 Здоровых собак</h3>
        <h2>{:.1f}%</h2>
    </div>
    """.format(healthy_pct), unsafe_allow_html=True)

with col3:
    avg_age = filtered_df['Возраст'].mean()
    st.markdown("""
    <div class="metric-container">
        <h3>🎂 Средний возраст</h3>
        <h2>{:.1f} лет</h2>
    </div>
    """.format(avg_age), unsafe_allow_html=True)

with col4:
    avg_weight = filtered_df['Вес (кг)'].mean()
    st.markdown("""
    <div class="metric-container">
        <h3>⚖️ Средний вес</h3>
        <h2>{:.1f} кг</h2>
    </div>
    """.format(avg_weight), unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Исследовательский анализ", "🎯 Модели машинного обучения", "🔮 Прогнозы здоровья", "📈 Интерактивные графики", "🎛️ Исследователь данных"])

with tab1:
    st.markdown('<div class="section-header">📊 Исследовательский анализ данных</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        health_by_breed = filtered_df.groupby(['Порода', 'Здоров']).size().unstack(fill_value=0)
        if not health_by_breed.empty and 'Да' in health_by_breed.columns:
            health_by_breed['Процент_здоровья'] = health_by_breed['Да'] / (
                        health_by_breed['Да'] + health_by_breed.get('Нет', 0)) * 100

            fig = px.bar(
                x=health_by_breed.index,
                y=health_by_breed['Процент_здоровья'],
                title="🏥 Процент здоровых собак по породам",
                color=health_by_breed['Процент_здоровья'],
                color_continuous_scale="RdYlGn"
            )
            fig.update_layout(
                xaxis_title="Порода",
                yaxis_title="Процент здоровых (%)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        size_counts = filtered_df['Размер породы'].value_counts().dropna()
        if not size_counts.empty:
            fig = px.pie(
                values=size_counts.values,
                names=size_counts.index,
                title="🐕 Распределение по размерам породы",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        activity_health = pd.crosstab(filtered_df['Уровень активности'], filtered_df['Здоров'],
                                      normalize='index') * 100
        if not activity_health.empty and 'Да' in activity_health.columns:
            fig = px.bar(
                activity_health,
                y=activity_health.index,
                x='Да',
                orientation='h',
                title="🏃 Уровень активности и процент здоровых собак",
                color='Да',
                color_continuous_scale="Viridis"
            )
            fig.update_layout(xaxis_title="Процент здоровых (%)", yaxis_title="Уровень активности")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        scatter_df = filtered_df.dropna(subset=['Возраст', 'Вес (кг)', 'Посещений ветеринара в год'])
        scatter_df = scatter_df[scatter_df['Посещений ветеринара в год'] >= 0]
        if not scatter_df.empty:
            fig = px.scatter(
                scatter_df,
                x='Возраст',
                y='Вес (кг)',
                color='Здоров',
                size='Посещений ветеринара в год',
                title="📈 Возраст и вес по статусу здоровья",
                color_discrete_map={'Да': '#2E8B57', 'Нет': '#DC143C'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Нет данных для построения диаграммы рассеяния из-за отсутствия полных данных.")

with tab2:
    st.markdown('<div class="section-header">🎯 Модели машинного обучения</div>', unsafe_allow_html=True)

    ml_df = filtered_df.dropna(subset=['Здоров']).copy()

    if len(ml_df) > 10:
        le_dict = {}
        feature_columns = ['Порода', 'Размер породы', 'Пол', 'Уровень активности', 'Диета', 'Статус кастрации']

        for col in feature_columns:
            if col in ml_df.columns:
                le = LabelEncoder()
                ml_df[col + '_encoded'] = le.fit_transform(ml_df[col].fillna('Неизвестно'))
                le_dict[col] = le

        X_columns = [col + '_encoded' for col in feature_columns if col in ml_df.columns] + ['Возраст', 'Вес (кг)',
                                                                                             'Посещений ветеринара в год']
        X = ml_df[X_columns].fillna(ml_df[['Возраст', 'Вес (кг)', 'Посещений ветеринара в год']].mean())
        y = (ml_df['Здоров'] == 'Да').astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🌳 Модель случайного леса")

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_pred)

            st.metric("Точность модели", f"{rf_accuracy:.3f}")

            importance_df = pd.DataFrame({
                'Признак': X.columns,
                'Важность': rf_model.feature_importances_
            }).sort_values('Важность', ascending=True)

            fig = px.bar(
                importance_df,
                x='Важность',
                y='Признак',
                orientation='h',
                title="Важность признаков в модели случайного леса",
                color='Важность',
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 Модель логистической регрессии")

            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X_train, y_train)
            lr_pred = lr_model.predict(X_test)
            lr_accuracy = accuracy_score(y_test, lr_pred)

            st.metric("Точность модели", f"{lr_accuracy:.3f}")

            cm = confusion_matrix(y_test, lr_pred)
            fig = px.imshow(
                cm,
                text_auto=True,
                title="Матрица ошибок логистической регрессии",
                color_continuous_scale="Blues",
                labels=dict(x="Предсказанный класс", y="Фактический класс")
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🏆 Сравнение моделей")
        comparison_df = pd.DataFrame({
            'Модель': ['Случайный лес', 'Логистическая регрессия'],
            'Точность': [rf_accuracy, lr_accuracy],
            'Тип': ['Ансамбль', 'Линейная']
        })

        fig = px.bar(
            comparison_df,
            x='Модель',
            y='Точность',
            color='Тип',
            title="Сравнение производительности моделей",
            text='Точность'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Недостаточно данных для обучения моделей. Пожалуйста, расширьте фильтры.")

with tab3:
    st.markdown('<div class="section-header">🔮 Инструмент прогнозирования здоровья</div>', unsafe_allow_html=True)

    if len(ml_df) > 10:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Введите данные о собаке:")

            pred_breed = st.selectbox("Порода собаки:", options=df['Порода'].unique())
            pred_size = st.selectbox("Размер породы:", options=df['Размер породы'].unique())
            pred_sex = st.selectbox("Пол собаки:", options=df['Пол'].unique())
            pred_age = st.slider("Возраст (лет):", min_value=1, max_value=18, value=5)
            pred_weight = st.slider("Вес (кг):", min_value=2, max_value=70, value=23)
            pred_activity = st.selectbox("Уровень активности:", options=df['Уровень активности'].unique())
            pred_diet = st.selectbox("Тип диеты:", options=df['Диета'].unique())
            pred_spay = st.selectbox("Статус кастрации:", options=df['Статус кастрации'].unique())
            pred_vet = st.slider("Посещений ветеринара в год:", min_value=0, max_value=5, value=1)

        with col2:
            if st.button("🔮 Спрогнозировать состояние здоровья", type="primary"):
                pred_data = {
                    'Порода': pred_breed,
                    'Размер породы': pred_size,
                    'Пол': pred_sex,
                    'Возраст': pred_age,
                    'Вес (кг)': pred_weight,
                    'Уровень активности': pred_activity,
                    'Диета': pred_diet,
                    'Статус кастрации': pred_spay,
                    'Посещений ветеринара в год': pred_vet
                }

                pred_encoded = []
                for col in feature_columns:
                    if col in le_dict:
                        try:
                            encoded_val = le_dict[col].transform([pred_data[col]])[0]
                        except ValueError:
                            encoded_val = 0
                        pred_encoded.append(encoded_val)

                pred_encoded.extend([pred_age, pred_weight, pred_vet])
                pred_array = np.array(pred_encoded).reshape(1, -1)

                rf_prob = rf_model.predict_proba(pred_array)[0][1]
                lr_prob = lr_model.predict_proba(pred_array)[0][1]
                avg_prob = (rf_prob + lr_prob) / 2

                prediction = "Здоров" if avg_prob > 0.5 else "Нездоров"

                box_class = "prediction-box" if prediction == "Здоров" else "prediction-box unhealthy-box"

                st.markdown(f"""
                <div class="{box_class}">
                    <h2>Прогноз: {prediction}</h2>
                    <h3>Уверенность: {avg_prob:.1%}</h3>
                    <p>Случайный лес: {rf_prob:.1%}</p>
                    <p>Логистическая регрессия: {lr_prob:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=avg_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Вероятность здоровья (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "lightgreen"},
                            {'range': [75, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">📈 Интерактивная визуализация данных</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        x_axis = st.selectbox("Выберите ось X:", ['Возраст', 'Вес (кг)', 'Посещений ветеринара в год'])
        y_axis = st.selectbox("Выберите ось Y:", ['Вес (кг)', 'Возраст', 'Посещений ветеринара в год'])

    with col2:
        color_by = st.selectbox("Цвет по параметру:", ['Здоров', 'Порода', 'Пол', 'Уровень активности'])
        chart_type = st.selectbox("Тип графика:", ['Диаграмма рассеяния', 'Диаграмма размаха', 'Скрипичная диаграмма'])

    if chart_type == 'Диаграмма рассеяния':
        scatter_df = filtered_df.dropna(subset=[x_axis, y_axis, 'Посещений ветеринара в год'])
        scatter_df = scatter_df[scatter_df['Посещений ветеринара в год'] >= 0]
        if not scatter_df.empty:
            fig = px.scatter(
                scatter_df,
                x=x_axis,
                y=y_axis,
                color=color_by,
                size='Посещений ветеринара в год',
                hover_data=['Порода', 'Возраст'],
                title=f"{x_axis} против {y_axis} с цветом по {color_by.lower()}"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Нет данных для построения диаграммы рассеяния из-за отсутствия полных данных.")
    elif chart_type == 'Диаграмма размаха':
        fig = px.box(
            filtered_df.dropna(subset=[y_axis]),
            x=color_by,
            y=y_axis,
            title=f"Распределение {y_axis.lower()} по {color_by.lower()}"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.violin(
            filtered_df.dropna(subset=[y_axis]),
            x=color_by,
            y=y_axis,
            title=f"Распределение {y_axis.lower()} по {color_by.lower()}"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔗 Матрица корреляции")
    numeric_cols = ['Возраст', 'Вес (кг)', 'Посещений ветеринара в год']
    corr_df = filtered_df[numeric_cols].corr()

    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        title="Матрица корреляции числовых признаков",
        color_continuous_scale="RdBu",
        labels=dict(x="Признаки", y="Признаки")
    )
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown('<div class="section-header">🎛️ Интерактивный исследователь данных</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        health_filter = st.multiselect(
            "Статус здоровья:",
            options=['Да', 'Нет'],
            default=['Да', 'Нет']
        )

    with col2:
        sex_filter = st.multiselect(
            "Пол собаки:",
            options=df['Пол'].unique(),
            default=df['Пол'].unique()
        )

    with col3:
        activity_filter = st.multiselect(
            "Уровень активности:",
            options=df['Уровень активности'].unique(),
            default=df['Уровень активности'].unique()
        )

    explore_df = filtered_df[
        (filtered_df['Здоров'].isin(health_filter + [None])) &
        (filtered_df['Пол'].isin(sex_filter)) &
        (filtered_df['Уровень активности'].isin(activity_filter))
    ]

    st.markdown(f"### 📋 Отфильтрованный набор данных ({len(explore_df)} строк)")

    csv = explore_df.to_csv(index=False)
    st.download_button(
        label="📥 Скачать данные как CSV",
        data=csv,
        file_name='отфильтрованные_данные_собак.csv',
        mime='text/csv',
    )

    page_size = st.slider("Строк на странице:", min_value=10, max_value=100, value=20)

    if len(explore_df) > 0:
        total_pages = len(explore_df) // page_size + (1 if len(explore_df) % page_size > 0 else 0)
        page = st.slider("Страница:", min_value=1, max_value=max(1, total_pages), value=1)

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        st.dataframe(
            explore_df.iloc[start_idx:end_idx],
            use_container_width=True,
            height=400
        )

        st.markdown(f"Показаны строки с {start_idx + 1} по {min(end_idx, len(explore_df))} из {len(explore_df)}")
    else:
        st.warning("Нет данных, соответствующих выбранным фильтрам.")