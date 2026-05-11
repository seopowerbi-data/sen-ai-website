# Reprise session sen-ai-website — 2026-05-11

## À lire AVANT de répondre

1. `~/.claude/projects/C--Users-leed-sen-ai-website/memory/MEMORY.md` (auto-loaded)
2. `project_todo_tracker.md` — section **"Session 2026-05-09/10/11 — bilan final"** en haut du fichier
3. `feedback_no_hardcoded_vertical.md` — règle clé : sen-ai est SaaS multi-vertical, ZÉRO hardcoded brand/competitor/vertical-specific dans code shared
4. `project_migration_seollm_to_aiscan.md` — invariants SaaS + check-list

## Bilan session précédente (record absolu : 14 commits)

**Phase C 5/5 ✅** (Position dist · Circuit breaker · Refund prorata · Gemini pool · Submodule bump)
**Phase A 7/9 ✅** (BrandResolver · migrations brand_promotion · Gemini pool · LlmUsageLog · brief · Kanban · Validation page)
**Phase B kickoff ✅** (FAQ handler + endpoint + Generate button + smoke E2E 100/100 sur PF own domain)
**2 bonus UX** (retry endpoint + permanent-failure detection)

Commits du jour : `594a56e` → `e75b72e` (15 commits, tous pushés + déployés prod).

## ⚠️ Ce qui marche PAS encore (point de reprise)

**Brand bias FAQ sur scan compétiteur** : architecturalement bloqué.

- Smoke test concret : FAQ generated on `uriage.com` (PF competitor) → output 100% Uriage products, 0 PF brand mentions, malgré workspace_brief PF + BrandResolver promu vers Avène + `target_site` patché.
- Cause : page scrape = page concurrent + web_search retourne URLs concurrent only + prompt seo-llm `"Cite UNIQUEMENT URLs vérifiées"` → le LLM n'a aucune URL Avène à citer.
- Fix structurel requis (au choix) :
  - **(a) Re-scoper opportunities** : sur scan compétiteur, `target_url` doit pointer vers la page USER qui répond à la même question, pas vers la page compétiteur. Plus propre conceptuellement mais demande de générer les opportunities différemment + un mapping intelligent (LLM peut suggérer une URL user pertinente).
  - **(b) Enrichir `_fetch_brand_context`** : web_search additionnel sur les sites des brands promues pour ajouter URLs PF dans `verified_urls`. Plus simple à coder, ~2-3h. Le LLM cite des URLs Avène ET résume la question, FAQ devient hybride (compare/contraste).

## Premier message à donner pour la nouvelle session

```
Reprise. Lis le tracker (section bilan final tout en haut) + feedback_no_hardcoded_vertical.
Hier on a livré 14 commits (Phase C complète + Phase A 7/9 + Phase B kickoff E2E validé).
Limitation identifiée : brand bias FAQ sur scan competitor ne marche pas (smoke test uriage.com
a généré FAQ 100% Uriage 0 PF). 2 options pour fix : (a) re-scoper opportunities pour target_url
= page USER, ou (b) enrichir _fetch_brand_context avec web_search sur sites brands promues.
Tu me proposes l'archi de chaque option avec trade-offs concrets, puis on choisit.
```

## État branche

`master` à `3fdebfb`, working tree propre (à part contenu untracked *dans* le submodule `worker/seo_llm`, sans impact repo parent).

Le chantier "16 fichiers non-commités" (audit-gratuit + homepage refresh + register fermé) a été rangé en 4 commits le 2026-05-11 : `8aca51c` chore gitignore/untrack, `0323976` registration kill-switch, `6e32a60` landing refresh, `3fdebfb` fix copy audit.

## Prod state

✅ 5 containers up (postgres / api / astro / worker / nginx)
✅ Worker boot clean avec 13 handlers (`generate_faq` registered, sub-module bumped)
✅ Toutes les features de la session déployées
✅ Tracker à jour avec gap docs honnête

## Reste à faire — par priorité

**Décision 2026-05-11** : Option A2 (manual pick `target_url` user-side) comme stepping stone. Long-term vision = 7 piliers UX dans `project_roadmap_content_port.md`. Tout patch livré = stepping stone, pas cul-de-sac (audit columns + UI placeholders).

| Item | Effort | Priorité | Pilier |
|---|---|---|---|
| **A2 — Manual target_url pick + audit column `target_url_source`** | ~5h | 🔴 Stratégique | 3 (foundation) |
| Wire `format_workspace_brief` dans classify_topics + autres analysers | ~30min | 🟡 | 1 |
| Credit debit/refund 1 content_credit/FAQ | ~1h | 🟡 | — |
| FAQ-specific validation UI (Q/R rows view) | ~1.5h | 🟡 | 5 (foundation) |
| Décider site.json (PF testimonial vs neutre) | discussion | 🟡 marketing | — |
| Tip.astro tooltips vertical-neutres | ~20min | 🟢 polish | — |
| Phase D — Sitemap index (débloque A1 auto-suggest + Pilier 4) | ~5j | 🟢 next big | 3 |
| Phase E élargie — Side-by-side validation + measurement loop | ~14j | 🟢 long-term | 5, 7 |
| Phase F — Voice fingerprint | ~7j | 🟢 long-term | 4 |
| Phase G — CMS integrations | ~10j | 🟢 long-term | 6 |
| APScheduler infra (Phase D + Phase E dep) | ~0.5j | 🟢 | — |
| Site type classifier `classify_citation_domains.py` | ~1j | 🟢 | — |

## ⚠️ Pièges à connaître

- `api/.env` peut perdre des vars silencieusement → toujours `diff api/.env api/.env.save` avant assumer qu'une clé est set
- Après `docker compose up -d api` qui recrée le container : TOUJOURS `docker compose restart nginx` sinon 502 (nginx cache l'ancienne IP)
- Submodule `worker/seo_llm` = vertical-locked seo_llm CLI Pierre Fabre. JAMAIS éditer dedans. Toujours wrapper côté SaaS avec stub / subclass pour découpler.
