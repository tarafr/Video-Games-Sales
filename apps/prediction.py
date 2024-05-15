import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import statsmodels.api as sm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@st.cache_data
def plot_regression_results(y_test, y_pred):
    # Функция для визуализации результатов линейной регрессии.
    # Построение графика фактических значений против предсказанных линейной регрессией
    fig = go.Figure()

    # Добавление истинных значений
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predicted vs Actual',
            marker=dict(color='blue', size=10, opacity=0.5)
        )
    )

    # Линия идеального прогноза
    fig.add_trace(
        go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Ideal Fit',
            line=dict(color='red', width=2)
        )
    )

    fig.update_layout(
        title='Actual vs. Predicted Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        legend_title='Type',
        xaxis=dict(showline=True),
        yaxis=dict(showline=True)
    )

    # Построение гистограммы ошибок
    errors = y_test - y_pred
    fig_errors = go.Figure()
    fig_errors.add_trace(
        go.Histogram(
            x=errors,
            nbinsx=50,
            marker_color='blue'
        )
    )
    fig_errors.update_layout(
        title='Distribution of Prediction Errors',
        xaxis_title='Prediction Error',
        yaxis_title='Frequency',
        xaxis=dict(showline=True),
        yaxis=dict(showline=True)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_errors, use_container_width=True)


@st.cache_data
def prepare_data(X, y):
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)
    return model, y_pred


@st.cache_data
def create_correlation_matrix(df):
    corr = df.corr().round(2)
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='bupu',
        annotation_text=corr.values
    )
    fig.update_layout(height=500)
    return fig


def calculate_metrics(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mse = mean_squared_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    return mape, mse, r_squared


def encode_features(df, features):
    encoders = {}
    for feature in features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        encoders[feature] = le
    return df, encoders


def app(df: pd.DataFrame, current_dir: Path):
    c1, c2 = st.columns(2)
    with c1:
        st.title("Анализ и прогнозирование продаж видеоигр")
        st.markdown("""
            На этой странице представлен анализ данных о продажах видеоигр и прогнозирование продаж на основе введенных параметров. 
            Используйте форму ниже для ввода параметров и выполнения прогноза.
        """)
    with c2:
        st.image(str(current_dir / 'images' / 'predict.webp'), use_column_width='auto')

    # Замена пропущенных значений в категориальных данных
    categorical_features = df.select_dtypes(include=['category', 'object']).columns.tolist()

    df_encoded = df.copy()

    # Применение LabelEncoding ко всем категориальным столбцам
    df, encoders = encode_features(df, categorical_features)
    # label_encoder = LabelEncoder()
    # for column in categorical_columns:
    #     df[column] = label_encoder.fit_transform(df[column])

    # Удаление колонок, не несущих значимой информации для анализа
    df = df.reindex(sorted(df.columns), axis=1)

    st.dataframe(df, use_container_width=True)
    df.drop(['Name', 'Rank'], axis=1, inplace=True)

    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Разделение данных                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.markdown("""
        ## Разделение и подготовка данных
        **Разделим данные** на обучающую и тестовую выборки, и определим переменные, которые используем в модели прогнозирования. 
        Для этого выберем количественные переменные, а также преобразуем категориальные переменные для включения их в модель.
    """)
    st.markdown("""
        ## Анализ корреляции между переменными
        Понимание корреляции между различными переменными помогает выявить, какие факторы наиболее сильно влияют на ценообразование.
    """)
    st.plotly_chart(create_correlation_matrix(df), use_container_width=True)

    # Для упрощения выберем несколько предикторов
    predictors = df.columns.drop('Global_Sales')
    target = 'Global_Sales'

    # Разделение данных на обучающую и тестовую выборки для месячных данных
    X_train, X_test, y_train, y_test = train_test_split(
        df[predictors],
        df[target],
        test_size=0.2,
        shuffle=False
    )
    tab1, tab2 = st.tabs(["Тренировочные данные", "Тестовые данные"])

    with tab1:
        st.subheader("Тренировочные данные")
        st.markdown("""
            **Описание:** Тренировочные данные используются для подгонки модели и оценки её параметров.
            Эти данные получены путем исключения из исходного датасета столбцов с временными метками и целевой переменной 'Global_Sales'.

            **Данные тренировочного набора (X_train)**.
            Обучающий набор данных содержит информацию о признаках, используемых для обучения модели.
        """)
        st.dataframe(X_train, use_container_width=True)
        st.markdown("""
            **Целевая переменная (y_train)**.
            Целевая переменная содержит значения цены, которые модель должна научиться прогнозировать.
            В качестве целевой переменной для тренировочного набора используются исключительно значения столбца 'Global_Sales'.
        """)
        st.dataframe(pd.DataFrame(y_train).T)

    with tab2:
        st.subheader("Тестовые данные")
        st.markdown("""
            **Описание:** Тестовые данные используются для проверки точности модели на данных, которые не участвовали в тренировке.
            Это позволяет оценить, как модель будет работать с новыми, ранее не виденными данными.
            """)
        st.markdown("""
            **Данные тестового набора (X_test)**.
            Тестовый набор данных содержит информацию о признаках, используемых для оценки модели.
        """)
        st.dataframe(X_test, use_container_width=True)
        st.markdown("""
            **Целевая переменная (y_test)**.
            Целевая переменная представляет собой значения, которые модель пытается предсказать.
        """)
        st.dataframe(pd.DataFrame(y_test).T)
    # # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # # ┃                     Множественная линейная регрессия                     ┃
    # # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.markdown(r"""
        ## Множественная линейная регрессия
        Множественная линейная регрессия позволяет оценивать зависимость одной зависимой переменной от двух или более независимых переменных. Это делает её отличным инструментом для анализа и прогнозирования, где несколько факторов влияют на интересующий результат.

        ### Формула множественной линейной регрессии

        Формула множественной линейной регрессии выглядит следующим образом:

        $$
        y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n + \varepsilon
        $$

        Где:
        - $ y $: Зависимая переменная (предсказываемая переменная). Это та переменная, значение которой мы пытаемся предсказать на основе независимых переменных.
        - $ \beta_0 $: Константа (интерцепт), представляющая собой значение $ y $, когда все независимые переменные равны нулю.
        - $ \beta_1, \beta_2, \ldots, \beta_n $: Коэффициенты независимых переменных, которые измеряют изменение зависимой переменной при изменении соответствующих независимых переменных.
        - $ x_1, x_2, \ldots, x_n $: Независимые переменные, используемые для предсказания значения $ y $.
        - $ \varepsilon $: Ошибка модели, описывающая разницу между наблюдаемыми значениями и значениями, предсказанными моделью.

        ### Описание параметров

        - **Зависимая переменная ( $ y $ )**: Это переменная, которую вы пытаетесь предсказать. Например, количество прокатов велосипедов может быть зависимой переменной, которую мы хотим предсказать на основе погоды, времени года и других условий.

        - **Константа ( $ \beta_0 $ )**: Это значение зависимой переменной, когда все входные (независимые) переменные равны нулю. В реальности это значение может не иметь физического смысла, особенно если ноль не является допустимым значением для независимых переменных.

        - **Коэффициенты ( $ \beta_1, \beta_2, \ldots, \beta_n $ )**: Эти значения указывают, насколько изменится зависимая переменная при изменении соответствующей независимой переменной на одну единицу, при условии что все остальные переменные остаются неизменными. Они являются ключевыми в понимании влияния каждой независимой переменной на зависимую переменную.

        - **Независимые переменные ( $ x_1, x_2, \ldots, x_n $ )**: Это переменные или факторы, которые предположительно влияют на зависимую переменную. В контексте вашего приложения это могут быть погода, день недели, сезон и другие.

        - **Ошибка модели ( $ \varepsilon $ )**: Ошибка модели показывает, насколько далеко наши предсказания от фактических значений. Это может быть вызвано неполным объяснением всех влияющих факторов или случайными изменениями, которые невозможно предсказать с помощью модели.
        
        ## Выбор переменных
        В качестве зависимых переменных рассматриваются:
        * Global_Sales: продажи по всему миру.
        
        В качестве независимых переменных рассматриваем:
        * Year: год выпуска игры;
        * Genre: жанр игры;
        * Platform: игровая платформа.
        * EU_Sales: продажи в Европе (в миллионах);
        * NA_Sales: продажи в Северной Америке (в миллионах);
        * JP_Sales: продажи в Японии (в миллионах);
        
        
        Каждая из этих переменных имеет значительное количество уникальных значений, что позволяет подробно анализировать влияние каждого аспекта на продажи игр. Эти переменные были выбраны на основе предположения, что разные платформы, жанры и временные периоды могут оказывать различное влияние на интерес покупателей и, соответственно, на продажи.
    """)

    linear_model, y_pred_train_linear = prepare_data(X_train, y_train)
    # Извлечение данных о параметрах модели
    st.subheader('Результаты модели')
    st.text(str(linear_model.summary())[:950])
    st.subheader('Коэффициенты модели')
    summary_data = linear_model.summary().tables[1]
    info = pd.DataFrame(summary_data.data[1:], columns=summary_data.data[0])
    st.dataframe(info, use_container_width=True, hide_index=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Прогноз на зависимые данные                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.markdown("""
        ## Анализ точности построения модели
        ## Прогноз на зависимые данные
    """)

    mape, mse, r_squared = calculate_metrics(y_train, y_pred_train_linear.values)

    st.info(f"""
        ### Результаты прогноза на зависимые данные
        - **MAPE (Средняя абсолютная процентная ошибка):** {mape:.2f}%
        - **MSE (Среднеквадратичная ошибка):** {mse:.2f}
        - **R² (Коэффициент детерминации):** {r_squared:.3f}
    """)
    plot_regression_results(y_train, y_pred_train_linear)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Прогноз на независимые данные                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    y_pred_test_linear = linear_model.predict(X_test)
    mape, mse, r_squared = calculate_metrics(y_test, y_pred_test_linear.values)

    st.info(f"""
        ### Результаты прогноза на нeзависимые данные
        - **MAPE (Средняя абсолютная процентная ошибка):** {mape:.2f}%
        - **MSE (Среднеквадратичная ошибка):** {mse:.2f}
        - **R² (Коэффициент детерминации):** {r_squared:.3f}
    """)
    plot_regression_results(y_test, y_pred_test_linear)

    st.markdown("""
        ## Анализ результатов модели линейной регрессии для прогнозирования глобальных продаж видеоигр
        
        ### Общий обзор результатов
        
        Модель линейной регрессии, разработанная для прогнозирования глобальных продаж видеоигр, показала высокую степень объяснения изменчивости данных с коэффициентом детерминации (R²) близким к 1 (0.997) как для обучающего, так и для тестового набора данных. Это указывает на очень хорошую подгонку модели к данным.
        
        Параметры модели:
        * Коэффициенты: Значимыми предикторами оказались NA_Sales, EU_Sales, JP_Sales, и Other_Sales, каждый из которых имеет положительное влияние на глобальные продажи. Особенно важны продажи в Северной Америке и Европе, которые имеют наибольший вес в модели.
        * Константа: Значение константы (интерцепт) модели близко к нулю и статистически не значимо, что указывает на низкое смещение при отсутствии продаж во всех регионах.
        
        Статистическая значимость:
        * Статистическая значимость коэффициентов: Большинство коэффициентов значимы, за исключением некоторых переменных, таких как Publisher, Platform, Year, и Genre, которые не оказывают значимого влияния на глобальные продажи в данной модели.
        * Мультиколлинеарность: Высокое значение условного числа (Cond. No.) может указывать на наличие мультиколлинеарности среди предикторов, что требует дополнительного анализа.
        
        Метрики качества модели
        * Среднеквадратическая ошибка (RMSE) и средняя абсолютная ошибка (MAE) находятся на очень низком уровне как для обучающего, так и для тестового набора данных, подтверждая высокую точность предсказаний модели.
        * R² для тестового набора данных также высок, что демонстрирует хорошую обобщающую способность модели.
        
        Выводы:
        * Модель линейной регрессии успешно аппроксимировала глобальные продажи видеоигр, показывая высокую точность предсказаний. Основной вклад в продажи вносят региональные продажи, в то время как другие факторы, такие как платформа, год выпуска и жанр, оказались менее значимыми. Рекомендуется провести дополнительный анализ для устранения потенциальной мультиколлинеарности и оптимизации модели.
   """)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Форма ввода                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    # print(X_test)
    with st.form("Ввод данных"):
        st.subheader('Введите параметры для прогноза')
        num_imputs = {}
        num_imputs['Year'] = st.number_input(
            "Год выпуска",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=int(df['Year'].mean())
        )
        num_imputs['NA_Sales'] = st.number_input(
            "Продажи в Северной Америке (млн)",
            min_value=0.0,
            max_value=200.0,
            step=0.1,
            value=df['NA_Sales'].mean(),
        )
        num_imputs['EU_Sales'] = st.number_input(
            "Продажи в Европе (млн)",
            min_value=0.0,
            max_value=200.0,
            step=0.1,
            value=df['EU_Sales'].mean()
        )
        num_imputs['JP_Sales'] = st.number_input(
            "Продажи в Японии (млн)",
            min_value=0.0,
            max_value=200.0,
            step=0.1,
            value=df['JP_Sales'].mean()
        )
        num_imputs['Other_Sales'] = st.number_input(
            "Продажи в других регионах (млн)",
            min_value=0.0,
            max_value=200.0,
            step=0.1,
            value=df['Other_Sales'].mean()
        )

        inputs = {}
        for column in categorical_features:
            selection = st.selectbox(f"Выберите {column}", options=df_encoded[column].unique(), index=0)
            inputs[column] = selection  # Вводим значения напрямую в текстовом виде

        if st.form_submit_button("Прогнозировать", type='primary', use_container_width=True):
            # Создаем строку с данными для декодирования
            encoded_inputs = {col: encoders[col].transform([inputs[col]])[0] for col in categorical_features}
            all_inputs = pd.Series({**encoded_inputs, **num_imputs})
            input_df = pd.DataFrame([all_inputs])
            input_df = input_df.reindex(sorted(X_test.columns), axis=1)
            prediction = max(linear_model.predict(input_df)[0], 0)
            st.success(f"Прогноз успешно выполнен! Прогнозируемые глобальные продажи: ${prediction:.2f} (млн)")

            fig = go.Figure(go.Indicator(
                mode="number",
                value=prediction.round(3),
                number={'prefix': "$", 'suffix': "(млн)"},
                title={"text": "Прогнозируемые глобальные продажи"}
            ))

            fig.update_layout(paper_bgcolor="#f0f2f6", font={'color': "purple", 'family': "Arial"})
            st.plotly_chart(fig, use_container_width=True)
