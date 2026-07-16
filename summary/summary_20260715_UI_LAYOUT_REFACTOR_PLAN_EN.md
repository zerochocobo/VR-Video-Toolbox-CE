# Unified left-navigation UI refactor plan

## Scan result

The home page in `main.py` already uses a left navigation rail with a right card-based content area. The following 11 functional modules still use `ttk.Notebook` as their primary layout:

1. `one_click/main.py` — one-click mosaic restoration (5 tabs)
2. `area_selection_rect_crop/main.py` — direct crop selection (4 tabs)
3. `area_selection_vr2flat/main.py` — selected-area VR to flat (4 tabs)
4. `tool_vr2flat/main.py` — VR to flat (3 tabs)
5. `tool_split_combine/main.py` — split/combine (2 tabs)
6. `tool_v360_trans/main.py` — projection conversion (2 tabs)
7. `tools/gui.py` — video utilities (7 tabs)
8. `tool_subtitle/gui.py` — subtitle tools (7 tabs)
9. `tool_si/gui.py` — simultaneous interpretation (4 tabs)
10. `tool_clonevoice/gui.py` — clone dubbing (4 tabs)
11. `tool_subembed/main.py` — VR hard-subtitle embedding (3 tabs)

Standalone windows such as `tool_subtitle/debug_analyzer.py` are not Notebook-based main screens and are out of scope for this pass.

## Execution order

1. Add shared `utils/ui_theme.py` with light/dark palettes, ttk style initialization, and a reusable left icon-navigation container. Navigation titles will be matched against multilingual keywords to select Segoe MDL2 icons, with a text-only fallback when the icon font is unavailable.
2. Move the home-page palette from `main.py` into the shared module and add a UI-theme selector to the global settings page. Persist the choice in `vr_toolbox_config.json` and rebuild the home page after switching.
3. Replace each module's Notebook with the shared left-navigation container one module at a time. Only container creation, page registration, and page-selection calls will change; controls, variables, callbacks, and business logic remain intact.
4. After each module, run syntax compilation and relevant tests, then create a dedicated commit.
5. Run the full test suite, scan for remaining Notebook layouts, and update the HANDOVER archive.

## Acceptance criteria

- Every former tab is reachable from the left navigation and the first page is selected by default.
- Existing `select(index)` calls remain valid; business callbacks and log-widget references are unchanged.
- The light theme preserves the current home-page appearance; the dark theme is selectable globally and inherited by subsequently opened modules.
- No processing logic, workflow, defaults, or file formats are changed.

## Implementation result

- All 11 modules now use `utils.ui_theme.SideNavigation` for their left-navigation/right-content layout.
- Light and dark theme smoke tests passed for the home page and every module; page counts and navigation switching are intact.
- Every English, Chinese, and Japanese tab title resolves to a non-default keyword-matched icon.
- No `ttk.Notebook` or `TNotebook.Tab` references remain in the codebase.
