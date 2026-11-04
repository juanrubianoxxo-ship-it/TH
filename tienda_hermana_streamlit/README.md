# Tienda Hermana — Streamlit

Aplicación para comparar un punto nuevo con tiendas abiertas de Book/JUN, diligenciar en paralelo la ficha del punto y la tienda hermana y estimar la venta con la lógica de la ficha anterior. La interfaz usa rojo, amarillo y azul inspirados en OXXO, sin incorporar su logotipo.

## Flujo de uso

1. Carga Book manualmente o configura su enlace de OneDrive en Streamlit Secrets. La app lee `JUN` y conserva únicamente registros `ESTADO = ABIERTA`.
2. Diligencia el nombre y las variables disponibles del punto nuevo. Los desplegables categóricos muestran valores existentes en Book. `TIE26` no se usa como atributo; `TE` se reserva para entrenar internamente el ranking de tiendas espejo.
3. Pulsa **Buscar tienda hermana**. La aplicación muestra el Top 10 estadístico; puedes escoger otra opción de la lista. Se muestra el CR/ID para distinguir tiendas con nombres repetidos.
4. En las dos fichas interactivas, completa **Tipo, Transparencia y Accesibilidad**. También puedes ingresar Estudiantes a 100 m, Estudiantes en el entorno/300 m y Celdas de parqueo, tanto para el punto nuevo como para la hermana. Esos datos se conservan en la captura y en el resumen; no se eliminan ni se reemplazan. Si se dejan vacíos, la aplicación los conserva como vacíos/opcionales.
5. Pulsa **Estimar venta**. La cifra principal aparece en miles de pesos para los 30 meses, junto con el promedio mensual, el rango P10–P90 y los vecinos usados por el modelo.

## Modelo estadístico y cálculo a 30 meses

La aplicación usa un único modelo estadístico: un regresor **KNN ponderado por distancia**. La distancia compara las variables históricas de Book/JUN por familias, normaliza las variables numéricas y trata las categóricas por coincidencia. Sus pesos se calibran con las relaciones históricas de `TE`; `TE` no se usa como variable predictora de ventas.

El vecino principal del modelo es la tienda hermana recomendada. Si el usuario elige otra del Top 10, esa tienda se fuerza dentro del conjunto de vecinos. La venta mensual se estima como la media ponderada por distancia de la variable disponible en Book: primero `VENTAS OUM` y, si no hay suficientes datos, `VU6M / 6` como promedio mensual de seis meses. El acumulado de 30 meses es `venta mensual estimada × 30`. Como Book no contiene una serie mensual completa, la app no inventa tendencia ni estacionalidad; P10 y P90 muestran la dispersión observada entre los vecinos.

Todos los campos existentes siguen presentes en los formularios. Los campos adicionales de Tipo, Transparencia, Accesibilidad, estudiantes y celdas de parqueo se conservan para la ficha y el resumen, pero no se convierten artificialmente en predictores históricos cuando Book no contiene observaciones equivalentes para entrenarlos.

## Conectar Book desde OneDrive

En Streamlit Cloud, abre **Settings / App settings → Secrets** y configura:

```toml
BOOK_ONEDRIVE_URL = "https://1drv.ms/x/s!TU_ENLACE_COMPARTIDO"
```

También se incluye `.streamlit/secrets.toml.example`. No pegues el enlace real en el código, README ni GitHub. El archivo local `.streamlit/secrets.toml` está excluido de `.gitignore`.

La app añade `download=1`, descarga el `.xlsx` durante la sesión y ofrece el botón **Actualizar Book desde OneDrive**. El enlace debe permitir descarga desde el servidor; si OneDrive exige login, se requiere autenticación aprobada por la organización (por ejemplo, Microsoft Graph). El enlace secreto no se muestra en pantalla.

## Ejecutar localmente

Requiere Python 3.10 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Para usar OneDrive localmente, crea `.streamlit/secrets.toml` a partir del ejemplo y agrega el enlace compartido. No subas ese archivo a GitHub. Ejecuta las pruebas con:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
