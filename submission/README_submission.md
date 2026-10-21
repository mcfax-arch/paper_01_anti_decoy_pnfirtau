# Как подать препринт: TechRxiv и Zenodo

## Вариант A. Zenodo (быстрее всего, DOI сразу)

1. Войдите на https://zenodo.org через **ORCID** (кнопка "Sign in with ORCID") — так авторство свяжется с 0009-0005-9238-3778.
2. Нажмите **New upload**.
3. Загрузите файлы:
   - `paper_en.pdf`
   - `submission/paper01_reproducibility_package.zip`
4. Заполните поля (готовые значения — в `submission/metadata.md`):
   - **Resource type:** Publication → Preprint
   - **Title:** (см. metadata.md)
   - **Creators:** Zhukov, Maxim; ORCID 0009-0005-9238-3778; affiliation Independent Research
   - **Description:** абстракт из `submission/abstract_full.txt` + строка про reproducibility package
   - **Keywords:** из metadata.md
   - **License:** CC BY 4.0
   - **Language:** English
   - **Version:** 1.0
   - **Publication date:** 2026-10-20
5. Нажмите **Publish** → Zenodo выдаст DOI вида `10.5281/zenodo.XXXXXXX`.
6. DOI можно заранее **Reserve DOI** (до публикации), чтобы вставить его в текст/письма.
7. Если позже выйдет версия 2 — загружайте как **New version**; concept DOI сохранится.

## Вариант B. TechRxiv (IEEE, препринт-сервер)

1. Зарегистрируйтесь на https://www.techrxiv.org (можно через ORCID).
2. **Submit** → тип **Preprint**.
3. Заполните:
   - Title, Abstract (короткая версия из `metadata.md`, если поле ограничено ~250 словами; иначе полная),
   - Authors: Maxim Zhukov, Independent Research, ORCID,
   - Keywords,
   - Subject area: **Computer Science / Machine Learning** (secondary: Artificial Intelligence),
   - License: CC BY 4.0 (или CC BY-NC-ND, если требуется),
   - Comments: "21 pages, 18 figures, 12 tables; reproducibility package included".
4. Загрузите:
   - `paper_en.pdf` (manuscript),
   - `submission/paper01_reproducibility_package.zip` (source/reproducibility files).
5. Подтвердите, что работа не опубликована и не рассматривается в другом месте (препринт-политика TechRxiv это допускает).
6. Отправьте на модерацию. После принятия TechRxiv присвоит DOI; его можно добавить в Zenodo/arXiv как related identifier.

## Что уже готово в папке `submission/`

- `metadata.md` — все поля для копирования (title, abstract, keywords, license, subject area).
- `abstract_full.txt` — полный абстракт в plain text.
- `zenodo.json` — метаданные в формате Zenodo API (если захотите загружать программно).
- `techrxiv_cover_note.md` — сопроводительная записка.
- `paper01_reproducibility_package.zip` — архив с исходниками, фигурами, результатами и скриптами.

## Важные оговорки

- **Не указывайте DOI заранее**, пока площадка его не выдала.
- **Не заявляйте "peer-reviewed"** — это препринт.
- **Не включайте** `.agency/`, `hf_cache/`, `checkpoints_c3/` — внутренние артефакты.
- Если площадка спросит про конфликт интересов/финансирование: "None declared" / "No funding".
- После публикации препринта можно подавать в журнал; большинство журналов допускают preprint.
