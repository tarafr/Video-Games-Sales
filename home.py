import streamlit as st
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


@st.cache_data
def get_data_info(df):
    info = pd.DataFrame()
    info.index = df.columns
    info['Тип данных'] = df.dtypes
    info['Уникальных'] = df.nunique()
    info['Количество значений'] = df.count()
    return info


@st.cache_data
def create_histogram(df, column_name):
    fig = px.histogram(
        df,
        x=column_name,
        marginal="box",
        title=f"Распределение {column_name}",
        template="plotly"
    )
    return fig


@st.cache_data
def create_correlation_matrix(df, features):
    corr = df[features].corr().round(2)
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='bupu',
        annotation_text=corr.values
    )
    fig.update_layout(height=500)
    return fig


@st.cache_data
def create_pairplot(df, selected_features, hue=None):
    sns.set_theme(style="whitegrid")
    pairplot_fig = sns.pairplot(
        df,
        vars=selected_features,
        hue=hue,
        palette='viridis',
        plot_kws={'alpha': 0.5, 's': 80, 'edgecolor': 'k'},
        height=3
    )
    plt.subplots_adjust(top=0.95)
    return pairplot_fig


def display_scatter_plot(df, numerical_features, categorical_features):
    from scipy.stats import stats
    c1, c2, c3, c4 = st.columns(4)
    feature1 = c1.selectbox('Первый признак', numerical_features, key='scatter_feature1')
    feature2 = c2.selectbox('Второй признак', numerical_features, index=2,
                            key='scatter_feature2')
    filter_by = c3.selectbox('Фильтровать по', [None, *categorical_features],
                             key='scatter_filter_by')

    correlation = round(stats.pearsonr(df[feature1], df[feature2])[0], 4)
    c4.metric("Корреляция", correlation)

    fig = px.scatter(
        df,
        x=feature1, y=feature2,
        color=filter_by, trendline='ols',
        opacity=0.5,
        template='plotly',
        title=f'Корреляция между {feature1} и {feature2}'
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_games_per_year(df):
    # Количество видеоигр по годам
    yearwisegame = df.groupby('Year')['Name'].count().reset_index()

    # Общее количество опубликованных игр за год
    fig = go.Figure(go.Bar(
        x=yearwisegame['Year'],
        y=yearwisegame['Name'],
        marker={'color': yearwisegame['Name'], 'colorscale': 'bupu'}
    ))
    fig.update_layout(
        title_text='Выпуск видеоигр по годам',
        xaxis_title="Год",
        yaxis_title="Количество выпущенных игр"
    )
    return fig


def plot_sales_by_genre_and_region(df):
    plt.figure(figsize=(15, 10))
    comp_table = pd.melt(df, id_vars=['Genre'], value_vars=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
                         var_name='Sale_Area', value_name='Sale_Price')
    comp_table.head()
    sns.barplot(x='Genre', y='Sale_Price', hue='Sale_Area', data=comp_table)
    st.pyplot(plt)


@st.cache_data
def plot_sales_by_region(df):
    EU = df.pivot_table('EU_Sales', columns='Name', aggfunc='sum').T
    EU = EU.sort_values(by='EU_Sales', ascending=False).iloc[0:5]
    EU_games = EU.index

    JP = df.pivot_table('JP_Sales', columns='Name', aggfunc='sum').T
    JP = JP.sort_values(by='JP_Sales', ascending=False).iloc[0:5]
    JP_games = JP.index

    NA = df.pivot_table('NA_Sales', columns='Name', aggfunc='sum').T
    NA = NA.sort_values(by='NA_Sales', ascending=False).iloc[0:5]
    NA_games = NA.index

    Other = df.pivot_table('Other_Sales', columns='Name', aggfunc='sum').T
    Other = Other.sort_values(by='Other_Sales', ascending=False).iloc[0:5]
    Other_games = Other.index

    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("North Americal", "Europe", "Japan", "Other"),
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5],
    )

    fig.add_trace(
        go.Bar(
            y=NA['NA_Sales'],
            x=NA_games,
            name="North America",
            marker={'color': NA['NA_Sales'], 'colorscale': 'Portland'}
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            y=EU['EU_Sales'],
            x=EU_games,
            name="Europe",
            marker={'color': EU['EU_Sales'], 'colorscale': 'Portland'},
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            y=JP['JP_Sales'],
            x=JP_games,
            name="Japan",
            marker={'color': JP['JP_Sales'], 'colorscale': 'Portland'},
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            y=Other['Other_Sales'],
            x=Other_games,
            name="Other",
            marker={'color': Other['Other_Sales'], 'colorscale': 'Portland'},
        ),
        row=2, col=2
    )
    fig.update_layout(height=800)
    return fig


@st.cache_data
def plot_sales_trends_by_region(df):
    regions_agg = {
        'NA_Sales': 'sum',
        'JP_Sales': 'sum',
        'EU_Sales': 'sum',
        'Other_Sales': 'sum',
        'Global_Sales': 'sum'
    }

    geo_tdf = df.groupby(['Year']).agg(regions_agg).reset_index()
    geo_tdf = geo_tdf.sort_values('Year', ascending=True)

    fig = go.Figure()
    for region in ['NA', 'JP', 'EU', 'Other']:
        fig.add_trace(go.Scatter(
            x=geo_tdf['Year'],
            y=geo_tdf[region + '_Sales'],
            mode='lines',
            name=region,
        ))
    fig.update_layout(
        title="Общий объем продаж в год по регионам (в миллионах)",
        xaxis_title="Год",
        yaxis_title="Сумма продаж",
        height=600
    )
    fig.update_xaxes(type='category')
    return fig


def app(df, current_dir: Path):
    cm = sns.light_palette("seagreen", as_cmap=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Главная страница                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.title("Анализ продаж видеоигр")
    st.markdown("## Область применения")
    markdown_col1, markdown_col2 = st.columns(2)

    markdown_col1.markdown(
        """
        В данном отчете представлен анализ датасета, содержащего информацию о продажах видеоигр с объемом продаж свыше 100,000 копий по всему миру. Эти данные были собраны с сайта vgchartz.com и содержат информацию о названии игр, платформах, на которых они выпускались, годе выпуска, жанре, издателе, а также региональных и глобальных продажах. Анализ этих данных позволит исследовать тренды и предпочтения в индустрии видеоигр, а также оценить влияние различных факторов на коммерческий успех игр.
        """
    )
    markdown_col2.image(str(current_dir / 'images' / 'main.webp'), use_column_width='auto')

    tab1, tab2 = st.tabs(["Описание данных", "Пример данных"])

    with tab1:
        st.markdown(
            r"""
            ## Ключевые параметры и характеристики
            С целью предоставления объективной аналитики в сфере разработки и выпуска видеоигр, а также содействия в улучшении качества игрового контента, было принято решение о выборе датасета, содержащего статистическую информацию о продажах видеоигр. В ходе данного проекта осуществляется анализ параметров, влияющих на успешность и приемлемость видеоигровых продуктов для потребителей. Целью данного исследования является не только выявление тенденций и факторов, влияющих на успех или неудачу игровых проектов, но и предоставление рекомендаций как крупным, так и независимым разработчикам с целью повышения качества создаваемого контента и удовлетворения запросов аудитории. Руководствуясь данными и выводами данного исследования, разработчики смогут принимать обоснованные решения в процессе создания и продвижения видеоигр, что, в конечном итоге, способствует развитию индустрии и улучшению опыта игроков.
            
            С каждым годом рынок видеоигр продолжает расширяться, вызывая соответствующий рост ожиданий у геймеров. AAA-релизы становятся центром внимания, ожидая от разработчиков инноваций и революционных изменений. Аналитические отделы таких компаний, как Activision, Bethesda Softworks, Square Enix и другие, активно исследуют рынок видеоигр, учитывая различные метрики, такие как выручка от продаж, активная аудитория, доход с одного игрока и показатели возврата инвестиций. Важно отметить, что аналитические отчеты разрабатываются не просто как документация, а как инструмент, помогающий принимать решения маркетологам, продюсерам и геймдизайнерам.
            
            К сожалению, некоторые крупные игровые компании, стремясь к максимизации прибыли, иногда теряют понимание сути геймдева. Это приводит к таким неудачным релизам, как "GTA: The Trilogy — The Definitive Edition", который в 2021 году стал объектом острого критицизма. Этот ремастер классической трёхмерной трилогии GTA стал известен своим низким качеством, включая баги, цензуру, низкую производительность, удаленные механики и ужасную графику. "Cyberpunk 2077" - еще один пример провального релиза, где амбиции разработчиков и маркетинговых стратегов столкнулись с реальностью. Несмотря на высокие ожидания и безупречную репутацию студии CD Project RED, игра столкнулась с критикой из-за массовых технических проблем и разочаровавшей игровой механики. В погоне за прибылью руководители игровых студий часто забывают о том, что именно делает игру незабываемой. Это базовые принципы геймдева, которые создают игры, становящиеся частью культуры.
            
            | Параметр                  | Описание                                                                       |
            |---------------------------|--------------------------------------------------------------------------------|
            |Rank|	Ранг по общим продажам|
            |Name|	Название игры|
            |Platform|	Платформа выпуска игры|
            |Year|	Год выпуска игры|
            |Genre|	Жанр игры|
            |Publisher|	Издатель|
            |NA_Sales|	Продажи в Северной Америке (млн копий)|
            |EU_Sales|	Продажи в Европе (млн копий)|
            |JP_Sales|	Продажи в Японии (млн копий)|
            |Other_Sales|	Продажи в других регионах мира (млн копий)|
            |Global_Sales|	Общемировые продажи (млн копий)|\n
    """)

    with tab2:
        st.header("Пример данных")
        st.dataframe(df.head(15))

    categorical_features = df.select_dtypes(include=['category', 'object']).columns.tolist()
    numerical_features = df.drop('Rank', axis=1).select_dtypes(include=['int64', 'float64']).columns.tolist()

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃               Предварительный анализ данных                 ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.header("Предварительный анализ данных")
    # Отображение метрик в колонках
    # Создание колонок для метрик
    col1, col2, col3 = st.columns(3)

    # Отображение метрик
    with col1:
        st.metric(label="Самая продаваемая игра", value=df.loc[df['Global_Sales'].idxmax()]['Name'])
        st.metric(label="Общий объем продаж (млн)", value=f"{df['Global_Sales'].sum():.2f}")

    with col2:
        st.metric(label="Наиболее популярный жанр", value=df['Genre'].mode()[0])
        st.metric(label="Средние продажи в NA (млн)", value=f"{df['NA_Sales'].mean():.2f}")

    with col3:
        st.metric(label="Самая популярная платформа", value=df['Platform'].mode()[0])
        st.metric(label="Средние продажи в EU (млн)", value=f"{df['EU_Sales'].mean():.2f}")

    st.dataframe(get_data_info(df), use_container_width=True)

    st.header("Основные статистики для признаков")

    tab1, tab2 = st.tabs(["Числовые признаки", "Категориальные признаки"])
    with tab1:
        st.header("Рассчитаем основные статистики для числовых признаков")
        st.dataframe(df.describe(), use_container_width=True)
    with tab2:
        st.header("Рассчитаем основные статистики для категориальных признаков")
        st.dataframe(df.describe(include=['category', 'object']), use_container_width=True)

    st.markdown("""
        Выводы по статистикам:
        * Продажи сильно варьируются, с низким средним и высоким стандартным отклонением, указывающим на большие выбросы в данных (например, крайне успешные игры).
        * Медианное значение значительно ниже среднего, что подтверждает наличие скошенности распределений в сторону низких значений с несколькими очень высокими.
        * Platform: 31 уникальная платформа, наиболее популярная - DS (2163 упоминания).
        * Year: 39 уникальных годов, наибольшее количество игр выпущено в 2009 году (1431 игра).
        * Genre: 12 уникальных жанров, наиболее распространенный - Action (3316 игр).
        * Publisher: 578 различных издателей, наиболее часто встречается Electronic Arts (1351 игра).
    """)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Визуализация                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.header("Визуализация числовых признаков")
    selected_feature = st.selectbox(
        "Выберите признак",
        numerical_features,
        key="create_histogram_selectbox"
    )
    hist_fig = create_histogram(
        df,
        selected_feature
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Корреляция                     ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.header("Корреляционный анализ")
    st.subheader("Распределение по различным признакам")
    display_scatter_plot(df, numerical_features, categorical_features)

    st.header('Рассчитаем корреляционную матрицу для числовых признаков')
    st.markdown("""
        Матрица корреляции позволяет определить связи между признаками. Значения в матрице колеблются от -1 до 1, где:
        - 1 означает положительную линейную корреляцию,
        - -1 означает отрицательную линейную корреляцию,
        - 0 означает отсутствие линейной корреляции.
    """)
    st.plotly_chart(create_correlation_matrix(df, numerical_features), use_container_width=True)
    st.markdown("""
        Матрица корреляции показывает степень линейной зависимости между различными продажами по регионам и общемировыми продажами, а также их связь с рангом игры. Ниже приведены ключевые выводы из анализа корреляции:
        
        * Rank и продажи: Отрицательные значения корреляции между рангом и продажами во всех регионах (NA_Sales, EU_Sales, JP_Sales, Other_Sales и Global_Sales) указывают на то, что игры с более высокими продажами имеют более низкий ранг (ближе к первому месту). Коэффициент корреляции варьируется от -0.43 для глобальных продаж до -0.27 для продаж в Японии, что говорит о наибольшем влиянии глобальных продаж на ранг.
        * Между региональными продажами: Корреляция между продажами в различных регионах (NA_Sales, EU_Sales, JP_Sales, Other_Sales) и глобальными продажами (Global_Sales) очень высока, особенно между NA_Sales и Global_Sales (0.94), что подчеркивает, что продажи в Северной Америке являются сильным индикатором глобального успеха игры. Схоже высокая корреляция между EU_Sales и Global_Sales (0.90).
        * Региональные взаимосвязи: Корреляция между продажами в Северной Америке (NA_Sales) и Европе (EU_Sales) также высока (0.77), указывая на похожие предпочтения или тенденции в этих регионах. Корреляция между JP_Sales и другими регионами заметно ниже, что может свидетельствовать о различиях в рыночных предпочтениях или культурных особенностях.
        * Зависимость других продаж: Other_Sales также демонстрируют высокую корреляцию с EU_Sales (0.73) и Global_Sales (0.75), подчеркивая их вклад в общемировые продажи.
    """)
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃                     Диаграммы                        ┃
    # ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    st.markdown("""
        # Визуализация данных
        ## Гистограммы и ящики с усами
        
        Гистограмма — это вид диаграммы, представляющий распределение числовых данных. Она помогает оценить плотность вероятности распределения данных. Гистограммы идеально подходят для иллюстрации распределений признаков, таких как возраст клиентов или продолжительность контакта в секундах.
        
        Ящик с усами — это еще один тип графика для визуализации распределения числовых данных. Он показывает медиану, первый и третий квартили, а также "усы", которые простираются до крайних точек данных, не считая выбросов. Ящики с усами особенно полезны для сравнения распределений между несколькими группами и выявления выбросов.
        
        ### Анализ числовых признаков
        
        NA_Sales (Продажи в Северной Америке):
        * Гистограмма: Большинство значений сосредоточено близко к нулю, указывая на низкие продажи большинства игр.
        * Ящик с усами: Наличие множества выбросов свидетельствует о наличии игр с очень высокими продажами.
        
        EU_Sales (Продажи в Европе):
        * Гистограмма: Аналогично продажам в Северной Америке, большинство игр имеет низкие продажи.
        * Ящик с усами: Выбросы указывают на наличие нескольких игр с значительно высокими продажами.
        
        JP_Sales (Продажи в Японии):
        * Гистограмма: Еще более скошенное распределение к нулю по сравнению с другими регионами.
        * Ящик с усами: Меньше выбросов, чем в других регионах, что может указывать на отличия в маркетинговой стратегии или предпочтениях.
        
        Other_Sales (Продажи в других регионах):
        * Гистограмма: Сильное скошенное распределение с малым количеством игр, имеющих значимые продажи.
        * Ящик с усами: Подтверждает наличие выбросов, хотя их меньше по сравнению с NA и EU.
        
        Global_Sales (Глобальные продажи):
        * Гистограмма: Показывает общую тенденцию в продажах игр по всему миру, аналогичную региональным продажам.
        * Ящик с усами: Выбросы здесь более заметны, подчеркивая успешность отдельных топовых игр.
""")

    filtered_data = df.copy()

    def filter_outliers(df, column_name):
        """Отфильтровать выбросы данных на основе метода IQR для указанного столбца."""
        Q3 = df[column_name].quantile(0.75)
        Q1 = df[column_name].quantile(0.25)
        IQR = Q3 - Q1
        upper = Q3 + (1.5 * IQR)
        lower = Q1 - (1.5 * IQR)
        filtered_df = df[(df[column_name] > lower) & (df[column_name] < upper)]
        return filtered_df

    for col in numerical_features:
        filtered_data = filter_outliers(filtered_data, col)

    # plot_numeric_columns(filtered_data, numerical_features)
    st.image(str(current_dir / 'images' / 'plot_numeric_columns.png'), use_column_width='auto')

    st.markdown("""
        ### Визуализации по годам выпуска
        Выпуск видеоигр по годам:
        * Пик выпуска новых игр пришелся на период с 2005 по 2010 годы, после чего наблюдается стабилизация и некоторое снижение активности.
        
        Продажи видеоигр в разбивке по годам выпуска:
        * Виден рост продаж с начала 1990-х до пика в 2000-х годах, что коррелирует с общим увеличением популярности видеоигр и технологическим прогрессом.
    """)
    st.plotly_chart(plot_games_per_year(df), use_container_width=True)

    st.markdown("""
        ## Рейтинг видеоигр по регионам
        Топ игр по регионам:
        * Северная Америка и Европа: Доминируют игры, такие как "Wii Sports" и "Grand Theft Auto V", показывая предпочтение экшн и спортивных жанров.
        * Япония: Лидируют "Pokemon" и "Super Mario", что отражает культурные различия в предпочтениях игр.
        * Другие регионы: Смесь жанров с некоторым преобладанием экшн-игр.
      """)

    plot_sales_by_genre_and_region(df)
    st.plotly_chart(plot_sales_by_region(df), use_container_width=True)

    st.markdown("""
        ### Общий Объем Продаж по Регионам
        Динамика продаж по регионам: Наблюдается пик продаж во всех регионах в период с 2005 по 2010 годы, что может быть связано с ростом популярности цифровых развлечений и увеличением доступности интернета. После 2010 года виден спад, вероятно, из-за насыщения рынка и изменений в потребительских предпочтениях.
    """)
    st.plotly_chart(plot_sales_trends_by_region(df), use_container_width=True)

    st.markdown(
        """
        ## Столбчатые диаграммы для категориальных признаков
        """
    )

    def viz1(df):
        sns.barplot(x='Genre', y='Global_Sales', data=df)
        plt.title('Global')
        st.pyplot(plt, use_container_width=True)

    viz1(df)

    st.markdown("""
       Похоже, что платформенные игры лидируют. За ними следуют стрелялки игры. Кроме того, кажется, что игры advanture - это наименее любимые игры, в которые можно играть.
       Давайте посмотрим на разные части мира с помощью этой игровой жанровой функции. Это тоже будет выглядеть так?
    """)

    def viz2(df):
        f, axes = plt.subplots(3, 1)
        plt.rcParams["figure.figsize"] = (18, 15)
        sns.barplot(x='Genre', y='NA_Sales', data=df, ax=axes[0]).set(title='NORTH AMERICA, EUROPE and JAPAN')
        sns.barplot(x='Genre', y='EU_Sales', data=df, ax=axes[1])
        sns.barplot(x='Genre', y='JP_Sales', data=df, ax=axes[2])
        st.pyplot(plt, use_container_width=True)

    viz2(df)
    st.markdown("""
        Цель анализа — выявить региональные различия в популярности жанров, что может помочь разработчикам и издателям оптимизировать стратегии маркетинга и разработки игр.
        
        Северная Америка:
        * Как вы можете видеть, на Северную Америку приходится почти половина мировых продаж.
        * Платформеры и спортивные игры показывают высокие продажи, что подчеркивает любовь к традиционным и динамичным играм.
        * Жанры, такие как стрелялки и платформеры, также популярны, подчеркивая предпочтение активных и захватывающих игровых процессов.
        
        Европа:
        * Европейский рынок больше склонен к стрелялкам и спортивным играм, подчеркивая интерес к конкурентным и стратегически сложным жанрам.
        * Относительно низкая популярность платформеров по сравнению с другими регионами может указывать на различные культурные предпочтения.
        
        Япония:
        * Ролевые игры значительно популярнее в Японии, чем в других регионах, что отражает культурную страсть к сюжетно-ориентированным и глубоко разработанным игровым мирам.
        * трелялки значительно менее популярны, что может быть связано с культурными различиями в восприятии жанра и предпочтениями в геймплее.
        
        Региональные различия: Жанровые предпочтения значительно различаются в зависимости от региона, что подчеркивает необходимость учета культурных особенностей при разработке и продвижении игр.
        Глобальные тенденции: Несмотря на региональные различия, определенные жанры, такие как спортивные и стрелялки, остаются популярными во всех регионах, что указывает на их универсальную привлекательность.
    """)
    st.markdown("""
        ### Какая игровая платформа наиболее популярна у людей?
        
        График глобальных продаж: PS2, X360, и PS3 лидируют по глобальным продажам, что подчеркивает их популярность и широкое принятие среди игроков. Интересно, что несмотря на технологическое преимущество новых платформ, как PS4, более старые системы, такие как PS2, всё еще демонстрируют выдающиеся результаты.
        
    """)

    def viz3(df):
        # Суммируем данные по платформам
        df_sorted = df.head(5000).groupby('Platform')['Global_Sales'].sum().reset_index().sort_values(by='Global_Sales',
                                                                                                      ascending=False)
        # Построение графика с использованием сгруппированных данных
        fig = px.bar(
            df_sorted,
            x='Platform',
            y='Global_Sales',
            title='Глобальные продажи лучших игровых платформ',
            labels={'Global_Sales': 'Global Sales (in millions)', 'Platform': 'Gaming Platform'},
            color='Platform',
            text='Global_Sales'
        )
        return fig

    st.plotly_chart(viz3(df), use_container_width=True)

    st.markdown(
        """
        ## Точечные диаграммы для пар числовых признаков
        Данный вид графика важен для оценки распределения каждого признака и исследования возможных взаимосвязей между различными переменными в данных о подержанных автомобилях. Вот подробное описание графиков и их значение в контексте выбранной области:
        """
    )
    selected_features = st.multiselect(
        'Выберите признаки',
        numerical_features,
        default=numerical_features,
        key='pairplot_vars'
    )

    # Опциональный выбор категориальной переменной для цветовой дифференциации
    hue_option = st.selectbox(
        'Выберите признак для цветового кодирования (hue)',
        ['None'] + categorical_features,
        index=0,
        key='pairplot_hue'
    )
    if hue_option == 'None':
        hue_option = None
    if selected_features:
        st.pyplot(create_pairplot(df, selected_features, hue=hue_option))
    else:
        st.error("Пожалуйста, выберите хотя бы один признак для создания pairplot.")
