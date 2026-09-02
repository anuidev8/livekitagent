# Notas de curación — data/ (base de conocimiento SETI para RAG)

**Fuente:** `Base_de_Conocimiento_SETI_Master_v7 1 (1).pdf` (20 páginas), provisto en el Escritorio del usuario.
**Fecha de curación:** 2026-09-02.

## Qué se mantuvo (páginas 1-9 y 18-20 del PDF)

El PDF original mezclaba dos dominios que el propio documento (página 10) marca como
"NO MEZCLAR": un dominio comercial/institucional y un dominio metodológico/operativo
interno. Se conservó únicamente el dominio comercial/institucional, repartido en:

- `01-identidad-posicionamiento.md` — identidad, propuesta de valor, Modelo PRIME, cultura, métricas, sedes, modelo organizacional, certificaciones de calidad (CMMI, ITIL, PMI, ISO, GPS).
- `02-portafolio-servicios-prime.md` — Desarrollo, DataSimple (datos/IA), Cloud, Prime OPS.
- `03-clientes-y-sectores.md` — portafolio de clientes por sector.
- `04-alianzas-partners.md` — AWS, Microsoft, Google Cloud, Oracle, MongoDB, IBM, Dynatrace, Grafana, Movizzon, Delphix, RedHat.
- `05-casos-de-exito.md` — casos cuantificados (BTG Pactual x2, Protección, Grupo Aval) + casos de éxito en IA del Centro de Excelencia (visión por computador en retail, agente autónomo de soporte TI).
- `06-talento-y-contacto.md` — talento humano/compensaciones, canales de contacto oficiales.

## Qué se excluyó a propósito (páginas 10-17 del PDF)

El "Manual Maestro Comercial y Metodológico — Ecosistema Integrado SETI-Spec & OpenSpec":
metodología interna de SETI para desarrollo de software asistido por IA (SDD, capas de
OpenSpec, flujos de trabajo Expandido/Abreviado, gobernanza de agentes, FAQ interno de
ese framework). Es documentación de proceso de ingeniería interno, no información que un
visitante de un kiosco corporativo deba recibir del guía de voz. Se excluyó por completo,
tal como pide la nota de configuración RAG del propio PDF de origen.

## Si se vuelve a generar este dataset

No copiar contenido de las páginas 10-17 del PDF fuente a `data/`. Si el PDF maestro se
actualiza, repetir esta misma separación antes de reconstruir el índice vectorial.
