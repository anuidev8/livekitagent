# Casos de Éxito — SETI S.A.S.

## Casos de éxito cuantificados (transformación TI e infraestructura)

- **BTG Pactual — Migración de Cargas Críticas:** Traslado exitoso de 78 servidores de misión crítica hacia la nube de AWS en un plazo estricto de 3 meses. Consiguió un ahorro recurrente comprobado de entre 7.000 y 8.000 USD mensuales mediante la estrategia de automatización ejecutada con la plataforma propietaria corporativa Timely. Reducción drástica de las métricas de RPO y RTO a tan solo 5 minutos.
- **BTG Pactual — Modernización Core BTGFX:** Diseño y despliegue integral en entorno de producción de 144 integraciones complejas de software en un periodo de 6 meses. Se realizó la migración depurativa de 70 tablas transaccionales masivas hacia Amazon RDS PostgreSQL operando bajo la arquitectura de contenedores de Amazon EKS, lo cual optimizó la velocidad de procesamiento global en un 40% y disminuyó de manera directa los costos globales de la plataforma en un 30%.
- **Protección S.A. — Eficiencia de Cómputo:** Formulación y ejecución de una estrategia avanzada de elasticidad de infraestructura dentro de Oracle Cloud Infrastructure (OCI). Se logró un ahorro presupuestal sistemático de entre el 37% y el 50% en los costos asociados a unidades de procesamiento de cómputo (ECPUs), acelerando en paralelo en un 15% la ejecución de consultas analíticas pesadas.
- **Grupo Aval — Continuidad de Negocio:** Suministro de soporte técnico permanente y administración elástica de infraestructuras críticas en entornos de topología multi-cloud. Mantenimiento impecable de una tasa de disponibilidad continua con cero incidentes de interrupción en los servicios transaccionales de cara al usuario final desde el año 2016.

## Casos de éxito en Inteligencia Artificial (Centro de Excelencia de Inteligencia Artificial y Datos — SETI)

### Sector Retail

Durante el desarrollo de iniciativas para el sector retail, el Centro de Excelencia de Inteligencia Artificial y Datos de SETI construyó tres demostradores tecnológicos enfocados en la optimización de la operación en tienda mediante técnicas de visión por computador e inteligencia artificial.

**1. Monitoreo Inteligente de Filas y Tiempos de Atención**

- *Desafío:* Las tiendas de retail requieren conocer en tiempo real cuándo se están formando filas en las cajas registradoras y cuánto tiempo permanecen los clientes esperando para ser atendidos, para optimizar la asignación de personal y mejorar la experiencia del cliente.
- *Solución implementada:* Aplicación en Python diseñada para ejecutarse en dispositivos Edge, capaz de analizar video proveniente de cámaras de seguridad y detectar automáticamente aglomeraciones de personas y tiempos de atención en cajas registradoras. Utiliza modelos de detección de personas basados en YOLO (Ultralytics) y un sistema de tracking para identificar individualmente a cada cliente. Permite definir zonas de interés para monitorear la formación de filas (conteo de personas, seguimiento individual mediante tracking, identificación automática de eventos de formación de filas, configuración dinámica de las áreas de monitoreo) y medir tiempos de atención (hora de ingreso y salida del cliente al área de atención, tiempo total de permanencia, identificador único generado por el sistema de tracking).
- *Tecnologías:* Python, Ultralytics YOLO, Computer Vision, PySide6, Tracking de objetos, Procesamiento Edge.
- *Valor generado:* Identificación temprana de congestiones, optimización de apertura de cajas, medición objetiva de tiempos de atención, ejecución local sin dependencia de la nube.

**2. Análisis de Interés de Clientes sobre Productos y Estanterías**

- *Desafío:* Conocer qué productos generan mayor interés dentro de una tienda es una tarea compleja, especialmente cuando se desea obtener información objetiva basada en el comportamiento real de los clientes.
- *Solución implementada:* Plataforma de análisis espacial que identifica las zonas de mayor interacción mediante visión artificial y analítica avanzada. La detección se realiza a través de Oracle Cloud Vision y procesamiento local en Python/OpenCV con transformación geométrica cenital. La metodología "Dwell Score" considera la cantidad de personas cercanas al estante, el tiempo de permanencia en la zona, la frecuencia de interacción observada y factores de penalización para evitar sesgos por proximidad a cajas.
- *Tecnologías:* Python, OpenCV, Oracle Cloud Vision, Transformaciones de perspectiva, Analítica espacial.
- *Valor generado:* Comprensión objetiva del comportamiento del cliente, optimización de distribución de productos, identificación de zonas de alto interés comercial, soporte para estrategias de merchandising.

**3. Identificación Automática de Espacios Vacíos en Estanterías**

- *Desafío:* Mantener niveles adecuados de abastecimiento requiere identificar rápidamente espacios vacíos o subutilizados dentro de las estanterías.
- *Solución implementada:* Aplicación web con tres modelos especializados: (1) detección de productos, (2) detección de estanterías y (3) detección de espacios vacíos. Exportados a ONNX para ejecución vía WebGPU en el navegador, reduciendo costos de infraestructura. El sistema calcula porcentaje de ocupación y dispone de un mapa de calor configurable con umbrales ajustables.
- *Tecnologías:* Python, ONNX, WebGPU, HTML, Google Cloud Platform, Computer Vision.
- *Valor generado:* Identificación automática de desabastecimiento, optimización del espacio comercial, procesamiento en el navegador del usuario, reducción de costos computacionales en la nube.

### Automatización Inteligente de Procesos

**4. Agente Autónomo para Atención y Gestión de Requerimientos de TI**

- *Desafío:* Los equipos de soporte reciben diariamente solicitudes repetitivas que consumen tiempo y generan retrasos. Además, muchas llegan con información incompleta, obligando a intercambios adicionales antes de iniciar la atención.
- *Solución implementada:* Sistema agéntico para la automatización del soporte L1, desarrollado en Python con LangChain y LangGraph, observabilidad vía LangSmith/LangStudio e integración con GLPI (ITSM) y Microsoft Teams. Casos de uso implementados: gestión automática de accesos a GitLab (crear, eliminar, validar, notificar), escalamiento inteligente a equipos especializados (infraestructura, RRHH, arquitectura), gestión del ciclo de vida de tickets (Abierto → En proceso → En espera → Finalizado).
- *Tecnologías:* Python, LangChain, LangGraph, LangSmith, LangStudio, GLPI, Microsoft Teams, Arquitecturas Multiagente, LLMs.
- *Valor generado:* Automatización de soporte L1, reducción de tiempos de atención, validación automática de datos, ejecución autónoma de accesos, escalamiento inteligente, trazabilidad completa.
