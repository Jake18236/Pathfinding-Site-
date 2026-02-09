---
layout: musing-post.njk
title: "Hello World: Welcome to Musings"
date: 2026-02-09
summary: "A first post to kick off the musings section — a place for thoughts on AI, teaching, and everything in between."
languages:
  - label: English
    code: en
    title: "Hello World: Welcome to Musings"
  - label: LLMese
    code: tok
    title: "toki a: kama pona tawa lipu Musings"
tags:
  - draft
  - musing
  - meta
---

<!-- lang:en -->

This is a sample blog post to get the Musings section started. Replace or delete this file and add your own markdown files to `src/musings/`.

## How to add a new post

Create a new `.md` file in `src/musings/` with front matter like this:

```yaml
---
layout: musing-post.njk
title: "Your Post Title"
date: 2026-03-15
summary: "A short teaser shown on the home page."
tags:
  - musing
  - your-tag
---
```

Then write your content in markdown below the front matter. Each post gets its own page automatically.

## How to make a bilingual post

Add a `languages` field to the front matter and use `<!-- lang:XX -->` comment markers to separate the two versions:

```yaml
languages:
  - label: English
    code: en
  - label: Español
    code: es
```

The first language listed is shown by default. The reader's choice is remembered across page loads.

<!-- lang:tok -->

Esta es una entrada de ejemplo para iniciar la sección de Musings. Reemplaza o elimina este archivo y agrega tus propios archivos markdown en `src/musings/`.

## Cómo agregar una nueva entrada

Crea un nuevo archivo `.md` en `src/musings/` con front matter como este:

```yaml
---
layout: musing-post.njk
title: "Título de tu entrada"
date: 2026-03-15
summary: "Un breve adelanto que se muestra en la página principal."
tags:
  - musing
  - tu-etiqueta
---
```

Luego escribe tu contenido en markdown debajo del front matter. Cada entrada obtiene su propia página automáticamente.

## Cómo hacer una entrada bilingüe

Agrega un campo `languages` al front matter y usa marcadores de comentario `<!-- lang:XX -->` para separar las dos versiones:

```yaml
languages:
  - label: English
    code: en
  - label: Español
    code: es
```

El primer idioma listado se muestra por defecto. La elección del lector se recuerda entre visitas.
