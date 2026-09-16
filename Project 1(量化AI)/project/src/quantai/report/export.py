"""导出模块：研报导出为 PDF / HTML。

PDF 导出优先用 weasyprint，不可用时降级为纯文本 PDF（reportlab）。
HTML 导出直接用 Plotly 的 write_html + 股评文本拼接。

设计选型理由：
- weasyprint 能完整渲染 Plotly 的交互式 HTML 转 PDF（含图表与样式），是首选；
  但它依赖 cairo/pango 等系统库，在精简环境可能装不上，所以必须有降级方案。
- reportlab 不依赖系统图形库，能保证「至少产出一个 PDF」，
  代价是无法嵌入 Plotly 图表，只能输出纯文字股评。
- 两套路径都失败时返回 None，由调用方决定是否提示用户安装依赖。
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from ..paths import EXPORT_DIR, ensure_dirs
from ..logger import get_logger

log = get_logger("export")


def export_html(fig, report: str, code: str, name: str = "") -> Path:
    """导出单股研报为 HTML（交互式图表 + 股评）。

    参数：
        fig:    Plotly Figure 对象，调用其 to_html 生成交互式图表片段。
        report: AI 股评文本。
        code:   股票代码，用于文件名与标题。
        name:   股票名称，用于标题展示，可选。

    返回：
        Path: 生成的 HTML 文件绝对路径。

    实现要点：
        - include_plotlyjs="cdn"：通过 CDN 加载 Plotly JS，减小文件体积；
          若需离线使用可改为 "inline"，但文件会显著变大。
        - full_html=False：只取图表 div，由本函数拼装完整 HTML，
          方便注入自定义样式与股评区块。
        - 文件名带时间戳，避免多次导出互相覆盖。
    """
    ensure_dirs()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"{code}_{ts}.html"
    # 只取图表片段，不生成完整 HTML，方便注入自定义结构
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{code} {name} 研报</title>
<style>body{{font-family:"Microsoft YaHei",sans-serif;max-width:1100px;margin:20px auto;padding:0 16px}}
h1{{color:#1976d2}}.report{{white-space:pre-wrap;line-height:1.8;font-size:15px}}</style>
</head><body>
<h1>{code} {name} 研报</h1>
<p>生成时间：{dt.datetime.now():%Y-%m-%d %H:%M:%S}</p>
{chart_html}
<h2>AI 股评</h2>
<div class="report">{report}</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    log.info("导出 HTML: %s", path)
    return path


def export_pdf(fig, report: str, code: str, name: str = "") -> Path | None:
    """导出研报为 PDF。优先 weasyprint，降级为 reportlab；都没有则返回 None。

    参数：
        fig:    Plotly Figure（weasyprint 路径下用于生成 HTML 再转 PDF）。
        report: AI 股评文本（两条路径都需要）。
        code:   股票代码，用于文件名与标题。
        name:   股票名称，可选。

    返回：
        Path | None: 成功返回 PDF 文件路径；两条路径都失败时返回 None。

    降级策略（两级）：
        1. weasyprint：把 fig 转成 HTML，再用 weasyprint 把 HTML 渲染成 PDF，
           能保留交互式图表的静态快照。ImportError 说明未安装；
           其他异常（如系统库缺失）记录 warning 后进入下一级。
        2. reportlab：纯文字 PDF，不依赖系统图形库。若连 reportlab 也未安装
           （ImportError），则返回 None，由调用方提示用户。
    """
    ensure_dirs()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"{code}_{ts}.pdf"
    # 尝试 weasyprint
    try:
        from weasyprint import HTML
        # 复用 export_html 产出的 HTML，weasyprint 直接把它渲染成 PDF
        html_path = export_html(fig, report, code, name)
        HTML(filename=str(html_path)).write_pdf(str(path))
        log.info("导出 PDF (weasyprint): %s", path)
        return path
    except ImportError:
        # 未安装 weasyprint，静默进入降级路径
        pass
    except Exception as exc:
        # 已安装但运行失败（常见于 cairo/pango 等系统库问题），记录后降级
        log.warning("weasyprint 导出失败：%s，尝试 reportlab", exc)
    # 降级为 reportlab
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm

        # 注册中文字体：依次尝试微软雅黑、黑体；都失败回退 Helvetica（中文会显示方块）
        # 优先 msyh.ttc 是因为 Win10/11 默认自带，覆盖率最高
        for font_name, font_path in [
            ("YaHei", "C:/Windows/Fonts/msyh.ttc"),
            ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
        ]:
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                cn_font = font_name
                break
            except Exception:
                cn_font = "Helvetica"

        doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()
        body_style = styles["BodyText"]
        body_style.fontName = cn_font
        title_style = styles["Heading1"]
        title_style.fontName = cn_font

        story = [Paragraph(f"{code} {name} 研报", title_style), Spacer(1, 0.5*cm)]
        story.append(Paragraph(f"生成时间：{dt.datetime.now():%Y-%m-%d %H:%M:%S}", body_style))
        story.append(Spacer(1, 0.5*cm))
        # 按换行拆分段落，逐段写入；转义尖括号防止被 reportlab 当作 XML 标签解析
        for para in report.split("\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para.replace("<", "&lt;").replace(">", "&gt;"), body_style))
                story.append(Spacer(1, 0.2*cm))
        doc.build(story)
        log.info("导出 PDF (reportlab): %s", path)
        return path
    except ImportError:
        # reportlab 也未安装，无法生成 PDF
        log.error("weasyprint 与 reportlab 均未安装，无法导出 PDF")
        return None
    except Exception as exc:
        # reportlab 运行时异常（字体注册失败等），同样返回 None
        log.error("PDF 导出失败：%s", exc)
        return None


def export_compare_html(fig, table: pd.DataFrame, ai_report: str, group1: str, group2: str) -> Path:
    """导出分组对比为 HTML。

    参数：
        fig:       Plotly 对比图（make_compare_chart 产出）。
        table:     对比数据表（compare_data 返回的 table），渲染为 HTML 表格。
        ai_report: AI 对比研报文本。
        group1/group2: 两组名称，用于标题与文件名。

    返回：
        Path: 生成的 HTML 文件路径。

    说明：
        - 与单股 export_html 相比，多了「对比数据表」区块，用 pandas to_html 生成，
          再注入内联 CSS 让表格有边框与斑马纹效果。
        - 文件名用 compare_ 前缀 + 时间戳，与单股研报区分开。
    """
    ensure_dirs()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORT_DIR / f"compare_{ts}.html"
    # 复用单股 HTML 的图表片段生成方式
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    # pandas to_html 直接产出带 <table> 标签的字符串，配合下方 CSS 类名控制样式
    table_html = table.to_html(index=False, classes="table", border=0)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{group1} vs {group2} 对比研报</title>
<style>body{{font-family:"Microsoft YaHei",sans-serif;max-width:1100px;margin:20px auto;padding:0 16px}}
h1{{color:#1976d2}}.table{{border-collapse:collapse;width:100%;margin:12px 0}}
.table th,.table td{{border:1px solid #ddd;padding:6px 10px;text-align:center}}
.table th{{background:#f5f5f5}}.report{{white-space:pre-wrap;line-height:1.8}}</style>
</head><body>
<h1>{group1} vs {group2} 对比研报</h1>
<p>生成时间：{dt.datetime.now():%Y-%m-%d %H:%M:%S}</p>
{chart_html}
<h2>对比数据表</h2>{table_html}
<h2>AI 对比研报</h2><div class="report">{ai_report}</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    return path
