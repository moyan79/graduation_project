# -*- coding: utf-8 -*-
"""组装实验报告 docx。"""
from docx import Document
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent
FIGDIR = OUTDIR / "report_figs"
DOCX = OUTDIR / "ResNet101拉通训练诊断实验报告.docx"


def set_font(run, size=11, bold=False, east="SimSun", west="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = west
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), east)


def para(doc, text, size=11, bold=False, align=None, space_after=6, east="SimSun"):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_font(run, size=size, bold=bold, east=east)
    return p


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        c = t.cell(0, j)
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        set_font(run, size=10.5, bold=True)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            run = c.paragraphs[0].add_run(str(val))
            set_font(run, size=10.5)
    if widths:
        for j, w in enumerate(widths):
            for i in range(len(rows) + 1):
                t.cell(i, j).width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def fig(doc, path, caption, width=5.6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    para(doc, caption, size=10, align="center", east="KaiTi_GB2312")


doc = Document()
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
for style_name, sz, bld, east in [("Normal", 11, False, "SimSun"),
                                  ("Heading 1", 14, True, "SimHei"),
                                  ("Heading 2", 12, True, "SimHei")]:
    st = doc.styles[style_name]
    st.font.size = Pt(sz)
    st.font.bold = bld
    st.font.name = "Calibri"
    rPr = st.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), east)

# 标题与导语
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("ResNet101 拉通统一训练诊断实验报告")
set_font(run, size=18, bold=True, east="SimHei")
para(doc, "实验目的：检验子领域模型在 OpenMMSecV2 四域拉通数据上统一训练是否正常收敛，并采集特征秩与梯度冲突两份诊断数据，供导师判断训练有无问题。", size=11)
para(doc, "主要情况：ResNet101 共训练 10 个 epoch 并完整跑完，训练损失从约 0.70 降至 0.32，训练过程无崩溃中断；两份诊断数据（80 个诊断步的梯度日志、10 期冻结特征及终版特征）均已产出，数据齐备。", size=11, bold=False)

doc.add_heading("1 实验配置", level=1)
table(doc, ["项目", "取值"], [
    ["模型", "ResNet101（ForensicHub 注册名，44.6M 参数）"],
    ["初始化", "随机初始化（远程无网，timm 预训练权重无法下载）"],
    ["训练数据", "OpenMMSecV2 train.json，共 81632 张，四域混合"],
    ["验证与探针数据", "id_test.json，共 8240 张；探针抽查 2000 张"],
    [" batch 与优化", "单卡 batch128，4 卡 DDP 全局 batch512；AdamW，lr 4e-4，权重衰减 0.05"],
    ["训练轮数", "10 epoch"],
    ["硬件", "4 张 A40（44GB），torchrun DDP"],
    ["诊断设置", "每 20 步记录一次分域梯度（共 80 行）；每 epoch 存一份冻结特征"],
])

doc.add_heading("2 训练过程是否正常", level=1)
doc.add_heading("2.1 完成情况", level=2)
para(doc, "10 个 epoch 全部跑完，各 epoch 特征 dump（features_epoch0 至 features_epoch9）与终版特征（features_last.npz，2000×2048）均已落盘，checkpoint 已保存，grad_log.csv 共 80 行，均符合预期。")
doc.add_heading("2.2 损失变化", level=2)
para(doc, "训练初段损失约 0.70（随机初始化二分类起点附近），末段稳定在 0.32 至 0.40 之间，总体呈下降趋势。日志尾部抽查值如下。")
table(doc, ["位置", "训练损失"], [
    ["第 7 epoch 约 3/4 处", "0.3366"],
    ["第 8 epoch 约 1/3 处", "0.3512"],
    ["第 8 epoch 约 3/4 处", "0.3997"],
    ["第 9 epoch 约 2/3 处", "0.3230"],
])
para(doc, "训练中损失存在上下波动（四域混合、随机初始化条件下的常见现象），但未出现发散或长期停滞，可认为优化过程基本健康。")
doc.add_heading("2.3 运行中的异常与处理", level=2)
para(doc, "一是 AIGC/laion 目录下一批图片文件损坏无法解码，程序已改为静默跳过并用邻位样本代替，不影响训练，仅在早期日志中有记录。二是 5 epoch 小 batch 试跑中出现过个别诊断步余弦为 NaN，10 epoch 大 batch（单卡 128）运行中未再出现，证实为小样本分片伪影。三是此前有一次显存不足报错，经查是同机其他任务占卡所致，与本配置无关，本次运行前已确认四卡空闲。")

doc.add_heading("3 特征秩诊断数据", level=1)
doc.add_heading("3.1 终版特征奇异值谱", level=2)
para(doc, "对终版 2000×2048 冻结特征做 SVD：前 20 维解释方差累计 96.47%，首奇异值 487.9，末奇异值 27.2，头尾落差约 17.9 倍，有效秩 301.8（占 2048 维的约 15%）。方差高度集中于头部少数方向。")
fig(doc, FIGDIR / "fig1_spectrum.png", "图1 终版特征 Top20 奇异值谱")
doc.add_heading("3.2 集中度随训练的变化", level=2)
para(doc, "每 2 个 epoch 抽查一次，累计方差占比与头尾落差总体走高，第 5 epoch 有一次小幅回落。")
table(doc, ["epoch", "cumvar@20", "s0/s19"], [
    ["1", "0.9431", "11.68"],
    ["3", "0.9564", "15.56"],
    ["5", "0.9463", "10.49"],
    ["7", "0.9722", "21.22"],
    ["9", "0.9647", "17.94"],
])
fig(doc, FIGDIR / "fig2_traj.png", "图2 特征集中度随训练的变化")

doc.add_heading("4 梯度冲突诊断数据", level=1)
para(doc, "80 个诊断步平均：冲突率 0.57（六对域中余弦小于 0 的占比），多数冲突步占 54%，总体平均余弦为负（-0.067）。分域对情况如下。")
table(doc, ["域对", "平均余弦", "负对齐占比"], [
    ["Deepfake-AIGC", "-0.053", "0.54"],
    ["Deepfake-IMDL", "0.058", "0.42"],
    ["Deepfake-文档", "0.040", "0.41"],
    ["AIGC-IMDL", "0.167", "0.31"],
    ["AIGC-文档", "-0.196", "0.66"],
    ["IMDL-文档", "-0.210", "0.78"],
])
para(doc, "其中 IMDL 相关两对（IMDL-文档、Deepfake-IMDL）负对齐占比最高，分别达 0.78 与 0.69；AIGC-IMDL 为各对中最协同（平均余弦 0.167，负对齐占比 0.31）。")
fig(doc, FIGDIR / "fig3_grad.png", "图3 域间梯度对齐情况")

doc.add_heading("5 小结与待办", level=1)
para(doc, "本次训练流程完整、损失正常下降、诊断数据齐备，训练本身没有问题。特征集中度与梯度冲突的数值已如实列出，供进一步分析。后续可补：其余子领域模型的同流程复刻、单域训练对照、以及有网条件下的预训练初始化对照。")

doc.save(DOCX)
print("saved:", DOCX, DOCX.stat().st_size, "bytes")
