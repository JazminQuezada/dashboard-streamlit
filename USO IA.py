import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Ventas",
    layout="wide"
)

# Título principal
st.title("Dashboard Análisis de Ventas")
st.markdown("---")


# Función para cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv('datos_dummies_ventas.csv')
    # Crear columna de fecha si no existe (usando índice como días)
    if 'Fecha' not in df.columns:
        from datetime import datetime, timedelta
        start_date = datetime(2022, 1, 1)
        df['Fecha'] = [start_date + timedelta(days=i) for i in range(len(df))]
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    return df


# Cargar datos
try:
    df = load_data()
    st.success(f"Datos cargados exitosamente: {len(df)} registros")

except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.stop()

# Sidebar para filtros
st.sidebar.header("🔍 Filtros")

# Filtro de fecha
fecha_min = df['Fecha'].min()
fecha_max = df['Fecha'].max()
fecha_rango = st.sidebar.date_input(
    "Rango de fechas",
    value=(fecha_min, fecha_max),
    min_value=fecha_min,
    max_value=fecha_max
)

# Filtro de producto
productos = st.sidebar.multiselect(
    "Seleccionar Productos",
    options=df['Producto'].unique(),
    default=df['Producto'].unique()
)

# Filtro de categoría
if 'Categoría' in df.columns:
    categorias = st.sidebar.multiselect(
        "Seleccionar Categorías",
        options=df['Categoría'].unique(),
        default=df['Categoría'].unique()
    )

    # Aplicar filtros con categoría
    df_filtrado = df[
        (df['Fecha'] >= pd.to_datetime(fecha_rango[0])) &
        (df['Fecha'] <= pd.to_datetime(fecha_rango[1])) &
        (df['Producto'].isin(productos)) &
        (df['Categoría'].isin(categorias))
        ]
else:
    # Aplicar filtros sin categoría
    df_filtrado = df[
        (df['Fecha'] >= pd.to_datetime(fecha_rango[0])) &
        (df['Fecha'] <= pd.to_datetime(fecha_rango[1])) &
        (df['Producto'].isin(productos))
        ]

# ====================
# SECCIÓN 1: TABLA DE DATOS
# ====================
st.header("Tabla de Datos")

# Mostrar tabla completa sin paginación ni búsqueda
st.dataframe(df_filtrado, use_container_width=True, height=400)
st.info(f"Total de registros: {len(df_filtrado)}")

st.markdown("---")

# ====================
# SECCIÓN 2: ESTADÍSTICAS DESCRIPTIVAS
# ====================
st.header("📈 Análisis Descriptivo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Venta Total", f"${df_filtrado['Precio_Total'].sum():,.2f}")
with col2:
    st.metric("Venta Promedio", f"${df_filtrado['Precio_Total'].mean():,.2f}")
with col3:
    st.metric("Cantidad Total", f"{df_filtrado['Cantidad'].sum():,}")
with col4:
    st.metric("Transacciones", f"{len(df_filtrado):,}")

# Estadísticas detalladas
st.subheader("Estadísticas Detalladas")
columnas_numericas = ['Cantidad', 'Precio_Unitario', 'Precio_Total']
stats_df = df_filtrado[columnas_numericas].describe()
st.dataframe(stats_df, use_container_width=True)

# Interpretación del análisis descriptivo
st.markdown("### Interpretación del Análisis Descriptivo")
st.markdown(f"""
**Resumen de Ventas:**
- Se registraron un total de **{len(df_filtrado)} transacciones** en el período analizado.
- Las ventas totales alcanzaron **${df_filtrado['Precio_Total'].sum():,.2f}**, con un ticket promedio de **${df_filtrado['Precio_Total'].mean():,.2f}** por transacción.
- Se vendieron **{df_filtrado['Cantidad'].sum():,} unidades** en total, con una cantidad promedio de **{df_filtrado['Cantidad'].mean():.1f} unidades** por transacción.

**Variabilidad de Precios:**
- El precio unitario promedio es de **${df_filtrado['Precio_Unitario'].mean():.2f}**, con un rango de **${df_filtrado['Precio_Unitario'].min():.2f}** a **${df_filtrado['Precio_Unitario'].max():.2f}**.
- La desviación estándar del precio total es de **${df_filtrado['Precio_Total'].std():.2f}**, lo que indica {"una alta variabilidad" if df_filtrado['Precio_Total'].std() > df_filtrado['Precio_Total'].mean() * 0.5 else "una variabilidad moderada"} en las ventas.

**Distribución de Datos:**
- El 50% de las transacciones tienen un valor inferior a **${df_filtrado['Precio_Total'].median():.2f}** (mediana).
- El 25% de las transacciones más altas superan los **${df_filtrado['Precio_Total'].quantile(0.75):.2f}**.
""")

st.markdown("---")

# ====================
# SECCIÓN 3: GRÁFICAS
# ====================
st.header("📊 Visualizaciones")

# Gráfica 1: Ventas por Producto
st.subheader("Ventas Totales por Producto")
ventas_producto = df_filtrado.groupby('Producto')['Precio_Total'].sum().sort_values(ascending=False)
fig1 = px.bar(
    x=ventas_producto.index,
    y=ventas_producto.values,
    labels={'x': 'Producto', 'y': 'Venta Total ($)'},
    title="Ventas por Producto",
    color=ventas_producto.values,
    color_continuous_scale='Blues'
)
fig1.update_layout(showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

# Interpretación Gráfica 1
st.markdown("#### Análisis de Ventas por Producto")
producto_top = ventas_producto.index[0]
venta_top = ventas_producto.values[0]
porcentaje_top = (venta_top / ventas_producto.sum()) * 100
producto_menor = ventas_producto.index[-1]
venta_menor = ventas_producto.values[-1]

st.markdown(f"""
**Hallazgos Clave:**
- **{producto_top}** es el producto más vendido con **${venta_top:,.2f}**, representando el **{porcentaje_top:.1f}%** de las ventas totales.
- **{producto_menor}** tiene las menores ventas con **${venta_menor:,.2f}**.
- Existe {"una distribución equilibrada" if ventas_producto.std() < ventas_producto.mean() * 0.3 else "una concentración significativa"} en las ventas entre productos.

**Recomendación:**
{"Considerar estrategias de promoción para los productos de menor venta y mantener el stock del producto líder." if porcentaje_top > 30 else "La distribución equilibrada sugiere una cartera de productos saludable."}
""")

st.markdown("---")

# Gráfica 2: Tendencia de Ventas en el Tiempo
st.subheader("Tendencia de Ventas en el Tiempo")
ventas_tiempo = df_filtrado.groupby(df_filtrado['Fecha'].dt.to_period('M'))['Precio_Total'].sum()
ventas_tiempo.index = ventas_tiempo.index.to_timestamp()
fig2 = px.line(
    x=ventas_tiempo.index,
    y=ventas_tiempo.values,
    labels={'x': 'Fecha', 'y': 'Venta Total ($)'},
    title="Evolución de Ventas Mensuales"
)
fig2.update_traces(line_color='#1f77b4', line_width=3)
st.plotly_chart(fig2, use_container_width=True)

# Interpretación Gráfica 2
st.markdown("#### Análisis de Tendencia Temporal")
mes_max = ventas_tiempo.idxmax()
venta_max = ventas_tiempo.max()
mes_min = ventas_tiempo.idxmin()
venta_min = ventas_tiempo.min()
crecimiento = ((ventas_tiempo.values[-1] - ventas_tiempo.values[0]) / ventas_tiempo.values[0]) * 100

st.markdown(f"""
**Patrón Temporal Identificado:**
- El mes con mayores ventas fue **{mes_max.strftime('%B %Y')}** con **${venta_max:,.2f}**.
- El mes con menores ventas fue **{mes_min.strftime('%B %Y')}** con **${venta_min:,.2f}**.
- La tendencia general muestra un {"crecimiento" if crecimiento > 0 else "decrecimiento"} del **{abs(crecimiento):.1f}%** entre el primer y último mes.

**Estacionalidad:**
{"Se observa variabilidad mensual que podría indicar patrones estacionales. Se recomienda análisis adicional para identificar temporadas altas." if ventas_tiempo.std() > ventas_tiempo.mean() * 0.2 else "Las ventas muestran relativa estabilidad a lo largo del tiempo."}

**Insight Estratégico:**
{"Planificar inventario y campañas de marketing enfocadas en los meses de mayor demanda." if crecimiento > 0 else "Implementar estrategias para revertir la tendencia decreciente."}
""")

st.markdown("---")

# Gráfica 3: Distribución de Ventas por Categoría (si existe)
st.subheader("Distribución de Ventas por Categoría")
if 'Categoría' in df_filtrado.columns:
    ventas_categoria = df_filtrado.groupby('Categoría')['Precio_Total'].sum()
    fig3 = px.pie(
        values=ventas_categoria.values,
        names=ventas_categoria.index,
        title="Participación de Ventas por Categoría",
        hole=0.4
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Interpretación Gráfica 3
    st.markdown("#### Análisis de Distribución por Categoría")
    categoria_top = ventas_categoria.idxmax()
    porcentaje_cat_top = (ventas_categoria.max() / ventas_categoria.sum()) * 100

    st.markdown(f"""
**Composición de Ventas:**
- **{categoria_top}** domina el mercado con el **{porcentaje_cat_top:.1f}%** de participación.
- Número total de categorías: **{len(ventas_categoria)}**

**Distribución por Categoría:**
{chr(10).join([f"- **{cat}**: ${val:,.2f} ({(val / ventas_categoria.sum()) * 100:.1f}%)" for cat, val in ventas_categoria.items()])}

**Conclusión:**
{"El portfolio está muy concentrado en una categoría. Considerar diversificación." if porcentaje_cat_top > 50 else "Existe una diversificación saludable entre categorías."}
""")
else:
    # Gráfica alternativa: Top 10 productos
    top_productos = df_filtrado.groupby('Producto')['Precio_Total'].sum().nlargest(10)
    fig3 = px.bar(
        x=top_productos.values,
        y=top_productos.index,
        orientation='h',
        labels={'x': 'Venta Total ($)', 'y': 'Producto'},
        title="Top 10 Productos más Vendidos"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### 🏆 Top 10 Productos - Análisis")
    st.markdown(f"""
**Productos Estrella:**
- Los 10 productos principales generan **${top_productos.sum():,.2f}** en ventas.
- Esto representa el **{(top_productos.sum() / df_filtrado['Precio_Total'].sum()) * 100:.1f}%** del total.

**Recomendación:**
Enfocar recursos de marketing y mantener disponibilidad de estos productos clave.
""")

st.markdown("---")

# ====================
# SECCIÓN 4: ANÁLISIS PREDICTIVO
# ====================
st.header("Análisis Predictivo")

st.info("Predicción de Ventas utilizando Regresión Lineal")

# Preparar datos para el modelo
df_modelo = df.copy()
df_modelo['Fecha_Num'] = (df_modelo['Fecha'] - df_modelo['Fecha'].min()).dt.days
df_modelo['Mes'] = df_modelo['Fecha'].dt.month
df_modelo['Año'] = df_modelo['Fecha'].dt.year

# Agrupar por día para predicción
df_diario = df_modelo.groupby('Fecha_Num').agg({
    'Precio_Total': 'sum',
    'Fecha': 'first'
}).reset_index()

# Verificar que hay suficientes datos
if len(df_diario) < 10:
    st.warning("No hay suficientes datos para realizar predicciones confiables")
    st.stop()

# Dividir datos
X = df_diario[['Fecha_Num']]
y = df_diario['Precio_Total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Métricas
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

col1, col2 = st.columns(2)
with col1:
    st.metric("R² Score", f"{r2:.4f}")
with col2:
    st.metric("RMSE", f"${rmse:,.2f}")

# Predicción futura (próximos 30 días)
ultimo_dia = df_diario['Fecha_Num'].max()
dias_futuros = np.array([[ultimo_dia + i] for i in range(1, 31)])
predicciones_futuras = modelo.predict(dias_futuros)

fechas_futuras = [df['Fecha'].max() + pd.Timedelta(days=i) for i in range(1, 31)]

# Visualizar predicción
st.subheader(" Predicción de Ventas (Próximos 30 días)")
fig4 = go.Figure()

# Datos históricos
fig4.add_trace(go.Scatter(
    x=df_diario['Fecha'],
    y=df_diario['Precio_Total'],
    mode='lines',
    name='Datos Históricos',
    line=dict(color='blue')
))

# Predicciones
fig4.add_trace(go.Scatter(
    x=fechas_futuras,
    y=predicciones_futuras,
    mode='lines+markers',
    name='Predicción',
    line=dict(color='red', dash='dash')
))

fig4.update_layout(
    title="Predicción de Ventas Diarias",
    xaxis_title="Fecha",
    yaxis_title="Venta Total ($)",
    hovermode='x unified'
)

st.plotly_chart(fig4, use_container_width=True)

# Tabla de predicciones
st.subheader("Tabla de Predicciones")
df_predicciones = pd.DataFrame({
    'Fecha': fechas_futuras,
    'Venta_Predicha': predicciones_futuras
})
df_predicciones['Venta_Predicha'] = df_predicciones['Venta_Predicha'].round(2)
st.dataframe(df_predicciones, use_container_width=True)

# Interpretación del Modelo Predictivo - MEJORADA
st.subheader("💡 Interpretación del Modelo Predictivo")

# Determinar calidad y tendencia
if r2 > 0.8:
    calidad = "Excelente"
    emoji_calidad = "⭐⭐⭐"
    interpretacion_r2 = "El modelo tiene un ajuste excepcional a los datos históricos."
elif r2 > 0.6:
    calidad = "Bueno"
    emoji_calidad = "⭐⭐"
    interpretacion_r2 = "El modelo tiene un ajuste aceptable, aunque hay margen de mejora."
elif r2 > 0.3:
    calidad = "Regular"
    emoji_calidad = "⭐"
    interpretacion_r2 = "El modelo captura algunas tendencias, pero tiene limitaciones."
else:
    calidad = "Bajo"
    emoji_calidad = "⚠️"
    interpretacion_r2 = "El modelo tiene dificultades para capturar el patrón de ventas. Se recomienda usar modelos más complejos."

if modelo.coef_[0] > 0:
    tendencia = "Creciente"
    emoji_tendencia = "📈"
    interpretacion_tendencia = "Las ventas muestran una tendencia al alza."
else:
    tendencia = "Decreciente"
    emoji_tendencia = "📉"
    interpretacion_tendencia = "Las ventas muestran una tendencia a la baja."

# Calcular estadísticas de predicción
prediccion_promedio = predicciones_futuras.mean()
prediccion_total_30dias = predicciones_futuras.sum()
venta_historica_promedio = df_diario['Precio_Total'].mean()
cambio_porcentual = ((prediccion_promedio - venta_historica_promedio) / venta_historica_promedio) * 100

# Mostrar interpretación completa
st.markdown(f"""
### Evaluación del Modelo de Regresión Lineal

**1. Calidad del Modelo ({calidad} {emoji_calidad})**

- **R² Score: {r2:.4f}**  
  {interpretacion_r2}  
  El modelo explica el **{r2 * 100:.2f}%** de la variabilidad en las ventas diarias.

- **RMSE: ${rmse:,.2f}**  
  Error promedio en las predicciones. Esto significa que las predicciones pueden desviarse aproximadamente **±${rmse:,.2f}** del valor real.  
  {"El error es relativamente alto comparado con el promedio de ventas." if rmse > venta_historica_promedio * 0.5 else "✅ El error es aceptable en relación al promedio de ventas."}

---

### 📈 Tendencia Identificada

**Tendencia: {tendencia} {emoji_tendencia}**

{interpretacion_tendencia}

- **Cambio diario promedio:** ${abs(modelo.coef_[0]):.2f} por día
- **Proyección de cambio:** {"Aumento" if modelo.coef_[0] > 0 else "Disminución"} de aproximadamente **${abs(modelo.coef_[0] * 30):,.2f}** en los próximos 30 días

---

### 🔮 Predicciones para los Próximos 30 Días

- **Venta diaria promedio histórica:** ${venta_historica_promedio:,.2f}
- **Venta diaria promedio predicha:** ${prediccion_promedio:,.2f}
- **Cambio esperado:** {'+' if cambio_porcentual > 0 else ''}{cambio_porcentual:.1f}%

- **Venta total esperada (30 días):** ${prediccion_total_30dias:,.2f}
- **Venta mínima esperada:** ${predicciones_futuras.min():,.2f}
- **Venta máxima esperada:** ${predicciones_futuras.max():,.2f}

---

### 💼 Recomendaciones Estratégicas

{"✅ **Aprovechar el momentum:** Con una tendencia creciente, es momento de invertir en marketing y aumentar el inventario." if modelo.coef_[0] > 0 else "⚠️ **Acción correctiva necesaria:** La tendencia decreciente requiere implementar estrategias de reactivación de ventas."}

{"✅ **Confiabilidad:** El modelo es confiable para la planificación a corto plazo." if r2 > 0.6 else "⚠️ **Precaución:** Debido al bajo R², use estas predicciones solo como referencia y complemente con otros análisis."}

**Próximos pasos sugeridos:**
1. Monitorear las ventas reales vs. predichas diariamente
2. {"Preparar inventario para el incremento esperado" if modelo.coef_[0] > 0 else "Analizar causas de la caída y ejecutar campañas promocionales"}
3. Actualizar el modelo cada semana con nuevos datos
4. Considerar factores externos (estacionalidad, eventos, competencia)
""")