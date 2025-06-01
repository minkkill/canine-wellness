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

# Настройка страницы
st.set_page_config(
    page_title="🐕 Дашборд анализа здоровья собак",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Пользовательские стили
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

# Заголовок
st.markdown('<h1 class="main-header">🐕 Дашборд анализа здоровья собак</h1>', unsafe_allow_html=True)

# Создание синтетических данных (расширенная версия)
@st.cache_data
def generate_dog_data():
    np.random.seed(42)

    breeds = ["Австралийская овчарка", "Такса", "Чихуахуа", "Сибирская хаски", "Боксер",
              "Лабрадор-ретривер", "Бульдог", "Золотистый ретривер", "Немецкая овчарка", "Пудель"]
    breed_sizes = ["Маленький", "Средний", "Крупный"]
    sexes = ["Самец", "Самка"]
    activity_levels = ["Отсутствует", "Низкая", "Умеренная", "Активная", "Очень активная"]
    diets = ["Сухой корм", "Влажный корм", "Домашняя еда", "Специальная диета", "Сырая диета"]
    spay_neuter = ["Не кастрирован", "Кастрирован", "Стерилизована"]

    data = []
    for i in range(1000):
        breed = np.random.choice(breeds)
        size = np.random.choice(breed_sizes) if np.random.random() > 0.05 else None
        sex = np.random.choice(sexes)
        age = np.random.normal(7, 3)
        age = max(1, min(18, age))

        # Вес зависит от размера породы
        if size == "Маленький":
            weight = np.random.normal(7, 4)  # перевод в кг
        elif size == "Средний":
            weight = np.random.normal(20, 7)
        elif size == "Крупный":
            weight = np.random.normal(32, 9)
        else:
            weight = np.random.normal(20, 11)

        weight = max(2.3, weight) if np.random.random() > 0.03 else None

        activity = np.random.choice(activity_levels)
        diet = np.random.choice(diets) if np.random.random() > 0.05 else None
        spay_neuter_status = np.random.choice(spay_neuter)
        vet_visits = np.random.poisson(1.5)

        # Логика для определения здоровья
        health_score = 0
        if age < 2:
            health_score += 20
        elif age < 8:
            health_score += 30
        else:
            health_score += 10

        if activity in ["Активная", "Очень активная"]:
            health_score += 25
        elif activity == "Умеренная":
            health_score += 15
        elif activity == "Низкая":
            health_score += 5

        if vet_visits >= 1: health_score += 20
        if spay_neuter_status in ["Кастрирован", "Стерилизована"]: health_score += 15
        if diet in ["Домашняя еда", "Специальная диета"]: health_score += 10

        healthy = "Да" if health_score + np.random.normal(0, 15) > 50 else "Нет"
        if np.random.random() < 0.02: healthy = None

        data.append({
            "Порода": breed,
            "Размер породы": size,
            "Пол": sex,
            "Возраст": round(age, 1),
            "Вес (кг)": round(weight, 1) if weight else None,
            "Уровень активности": activity,
            "Диета": diet,
            "Статус кастрации": spay_neuter_status,
            "Посещений ветеринара в год": vet_visits,
            "Здоров": healthy
        })

    return pd.DataFrame(data)

# Загрузка данных
df = generate_dog_data()

# Боковая панель
st.sidebar.markdown("## 🎛️ Панель управления")
st.sidebar.markdown("---")

# Фильтры
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
        <h3>📊 Всего</h3>
        <h2>{}</h2>
    </div>
    """.format(len(filtered_df)), unsafe_allow_html=True)

with col2:
    healthy_pct = (filtered_df['Здоров'] == 'Да').sum() / len(filtered_df.dropna(subset=['Здоров'])) * 100
    st.markdown("""
    <div class="metric-container">
        <h3>💚 Здоровых</h3>
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
    ["📊 Исследовательский анализ", "🎯 Модели МО", "🔮 Прогнозы", "📈 Интерактивные графики", "🎛️ Исследователь данных"])

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
                title="🐕 Распределение по размерам",
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
                title="🏃 Уровень активности vs Процент здоровых",
                color='Да',
                color_continuous_scale="Viridis"
            )
            fig.update_layout(xaxis_title="Процент здоровых (%)", yaxis_title="Уровень активности")
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(
            filtered_df.dropna(subset=['Возраст', 'Вес (кг)']),
            x='Возраст',
            y='Вес (кг)',
            color='Здоров',
            size='Посещений ветеринара в год',
            title="📈 Возраст vs Вес по статусу здоровья",
            color_discrete_map={'Да': '#2E8B57', 'Нет': '#DC143C'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

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
        X_columns = [col for col in X_columns if col in ml_df.columns or col.replace('_encoded', '') in ml_df.columns]

        X = ml_df[['Возраст', 'Вес (кг)', 'Посещений ветеринара в год'] + [col + '_encoded' for col in feature_columns if
                                                                  col in ml_df.columns]].fillna(
            ml_df[['Возраст', 'Вес (кг)', 'Посещений ветеринара в год']].mean())
        y = (ml_df['Здоров'] == 'Да').astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🌳 Модель случайного леса")

            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_accuracy = accuracy_score(y_test, rf_pred)

            st.metric("Точность", f"{rf_accuracy:.3f}")

            importance_df = pd.DataFrame({
                'Признак': X.columns,
                'Важность': rf_model.feature_importances_
            }).sort_values('Важность', ascending=True)

            fig = px.bar(
                importance_df,
                x='Важность',
                y='Признак',
                orientation='h',
                title="Важность признаков (Случайный лес)",
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

            st.metric("Точность", f"{lr_accuracy:.3f}")

            cm = confusion_matrix(y_test, lr_pred)
            fig = px.imshow(
                cm,
                text_auto=True,
                title="Матрица ошибок (Логистическая регрессия)",
                color_continuous_scale="Blues",
                labels=dict(x="Предсказано", y="Фактически")
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
        st.warning("Недостаточно данных для обучения модели. Пожалуйста, расширьте фильтры.")

with tab3:
    st.markdown('<div class="section-header">🔮 Инструмент прогнозирования здоровья</div>', unsafe_allow_html=True)

    if len(ml_df) > 10:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Введите данные о собаке:")

            pred_breed = st.selectbox("Порода:", options=df['Порода'].unique())
            pred_size = st.selectbox("Размер:", options=['Маленький', 'Средний', 'Крупный'])
            pred_sex = st.selectbox("Пол:", options=['Самец', 'Самка'])
            pred_age = st.slider("Возраст (лет):", min_value=1, max_value=18, value=5)
            pred_weight = st.slider("Вес (кг):", min_value=2, max_value=70, value=23)
            pred_activity = st.selectbox("Уровень активности:",
                                         options=['Отсутствует', 'Низкая', 'Умеренная', 'Активная', 'Очень активная'])
            pred_diet = st.selectbox("Диета:",
                                     options=['Сухой корм', 'Влажный корм', 'Домашняя еда', 'Специальная диета', 'Сырая диета'])
            pred_spay = st.selectbox("Статус кастрации:", options=['Не кастрирован', 'Кастрирован', 'Стерилизована'])
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
        color_by = st.selectbox("Цвет по:", ['Здоров', 'Порода', 'Пол', 'Уровень активности'])
        chart_type = st.selectbox("Тип графика:", ['Диаграмма рассеяния', 'Диаграмма размаха', 'Скрипичная диаграмма'])

    if chart_type == 'Диаграмма рассеяния':
        fig = px.scatter(
            filtered_df.dropna(subset=[x_axis, y_axis]),
            x=x_axis,
            y=y_axis,
            color=color_by,
            size='Посещений ветеринара в год',
            hover_data=['Порода', 'Возраст'],
            title=f"{x_axis} vs {y_axis} с цветом по {color_by}"
        )
    elif chart_type == 'Ящичная диаграмма':
        fig = px.box(
            filtered_df.dropna(subset=[y_axis]),
            x=color_by,
            y=y_axis,
            title=f"Распределение {y_axis} по {color_by}"
        )
    else:
        fig = px.violin(
            filtered_df.dropna(subset=[y_axis]),
            x=color_by,
            y=y_axis,
            title=f"Распределение {y_axis} по {color_by}"
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
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown('<div class="section-header">🎛️ Интерактивный исследователь данных</div>', unsafe_allow_html=True)

    # Опции фильтрации данных
    col1, col2, col3 = st.columns(3)

    with col1:
        health_filter = st.multiselect(
            "Статус здоровья:",
            options=['Да', 'Нет'],
            default=['Да', 'Нет']
        )

    with col2:
        sex_filter = st.multiselect(
            "Пол:",
            options=df['Пол'].unique(),
            default=df['Пол'].unique()
        )

    with col3:
        activity_filter = st.multiselect(
            "Уровень активности:",
            options=df['Уровень активности'].unique(),
            default=df['Уровень активности'].unique()
        )

    # Применение фильтров
    explore_df = filtered_df[
        (filtered_df['Здоров'].isin(health_filter + [None])) &
        (filtered_df['Пол'].isin(sex_filter)) &
        (filtered_df['Уровень активности'].isin(activity_filter))
        ]

    # Показать отфильтрованные данные
    st.markdown(f"### 📋 Отфильтрованный набор данных ({len(explore_df)} строк)")

    # Добавить кнопку загрузки
    csv = explore_df.to_csv(index=False)
    st.download_button(
        label="📥 Скачать отфильтрованные данные как CSV",
        data=csv,
        file_name='отфильтрованные_данные_собак.csv',
        mime='text/csv',
    )

    # Отображение данных с пагинацией
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