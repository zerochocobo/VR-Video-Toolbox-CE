# 首页左右结构统一改造计划

## 扫描结论

首页 `main.py` 已使用左侧导航栏、右侧卡片内容区的布局，并集中定义了浅色调色板。内部仍使用 `ttk.Notebook` 的功能模块共 11 个：

1. `one_click/main.py`：一键去马赛克（5 个页签）
2. `area_selection_rect_crop/main.py`：选区直接裁剪（4 个页签）
3. `area_selection_vr2flat/main.py`：选区 VR 转平面（4 个页签）
4. `tool_vr2flat/main.py`：VR 转平面（3 个页签）
5. `tool_split_combine/main.py`：分屏/合并（2 个页签）
6. `tool_v360_trans/main.py`：投影转换（2 个页签）
7. `tools/gui.py`：视频小工具箱（7 个页签）
8. `tool_subtitle/gui.py`：字幕工具（7 个页签）
9. `tool_si/gui.py`：同声传译（4 个页签）
10. `tool_clonevoice/gui.py`：克隆配音（4 个页签）
11. `tool_subembed/main.py`：VR 硬字幕嵌入（3 个页签）

`tool_subtitle/debug_analyzer.py` 等独立窗口不是 Notebook 主界面，暂不改变其功能结构。

## 实施顺序

1. 新增共享 `utils/ui_theme.py`：提供浅色/深色调色板、ttk 样式初始化，以及可复用的左侧图标导航容器。导航标题通过中英日关键词匹配 Segoe MDL2 图标，缺少图标字体时自动回退为文字布局。
2. 将 `main.py` 的首页调色板迁移到共享模块，并在全局设置页增加 UI 主题下拉框；主题写入 `vr_toolbox_config.json`，切换后重建首页。
3. 按模块逐个替换 Notebook 为共享左右导航容器。只替换容器创建、页签注册和选中页调用，保留原有控件、变量、回调和业务逻辑。
4. 每个模块完成后执行语法编译与相关测试，确认无误后单独提交 commit。
5. 全部完成后执行全量测试、扫描残留 Notebook，并更新 HANDOVER 归档。

## 验证标准

- 所有原有页签均可通过左侧导航切换，默认显示第一个页面。
- 原有 `select(index)` 调用继续有效，业务回调和日志控件引用不变。
- 浅色主题保持当前首页视觉；深色主题可在首页设置中启用并被后续打开的模块继承。
- 不修改逻辑模块、处理流程、参数默认值或文件格式。

## 实施结果

- 已完成 11 个模块的左右结构改造，所有页面均由 `utils.ui_theme.SideNavigation` 承载。
- 已完成亮/暗主题的首页及内部模块双主题烟测；所有页面数量和导航切换正常。
- 中英日所有 Tab 标题均可通过关键词匹配到非默认图标。
- 当前代码中已无 `ttk.Notebook` 或 `TNotebook.Tab` 残留。
