# iiif-dance-exhibit

A Hugo-powered IIIF exhibit showcasing Philadelphia's dance heritage through photographs from Temple University Libraries' collections, featuring the Pennsylvania Ballet and Philadelphia Dance Company (Philadanco).

## Collections Featured

- **Philadelphia Dance Collection** (p16002coll17) - Pennsylvania Ballet photographs from 1966-1997
- **George D. McDowell Philadelphia Evening Bulletin Photographs** (p15037coll3) - Philadanco and dance documentation from 1975-1981

## Themes

- Classical Ballet
- American Dance
- Contemporary Dance
- African American Dance
- Training & Rehearsal
- Productions
- Performers

## Run locally

```bash
hugo server -D
```

Or with full path:

```bash
hugo server -D --source /Users/alexwermer-colan/Code/Hawc2/iiif-dance-exhibit
```

## Data

Dance photograph records are stored in `data/dance.json`, with IIIF image service URLs derived from ContentDM manifest IDs. All images are served via the IIIF Image API.

## Deployment

This site deploys automatically to GitHub Pages via GitHub Actions when changes are pushed to the `main` branch.

## Credits

Images courtesy of Temple University Libraries, Special Collections Research Center.
