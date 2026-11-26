# LaTeX template — pwr-control documents

Shared LaTeX settings and title page used by the documents of this repository.

## Files

| File | Purpose |
|---|---|
| `settings.tex` | Packages, page geometry, header/footer, theorem environments, `mybox`, SI units, colors. Loaded once; a second `\input` is ignored. |
| `titlepage.tex` | Title page built from the `\Doc...` macros defined by the document, including the revision history table. |
| `pwr-control_logo_doc.png` | Logo placed in the header of every page. |
| `frontespizio_v2.tex`, `frontespizio_davide_ahmet.tex`, `pdf_output_settings.tex` | Older title pages kept for reference. |

## Usage

Documents live in `repos_doc/<name>/` or `research/<name>/`, two levels below the
repository root, so the template is reached with `../../AAA_template_latex_settings/`.

```latex
\documentclass[11pt,a4paper]{scrartcl}
\input{../../AAA_template_latex_settings/settings}

\newcommand{\DocCategory}{Technical Research Report}
\newcommand{\DocField}{Power Electronics and Control Systems}
\newcommand{\DocTitle}{%
    {\fontsize{20}{24}\selectfont\bfseries Main title:} \\[16pt]
    {\fontsize{16}{20}\selectfont\bfseries subtitle of the title.}
}
\newcommand{\DocSubtitle}{One line describing the document.}
\newcommand{\DocAuthor}{Davide Bagnara}
\newcommand{\DocRevision}{00}
\newcommand{\DocDate}{Month Year}
\newcommand{\DocAbstract}{Abstract of the document.}

% one line per revision: Rev. & Date & Description & Authors \\
\newcommand{\RevisionHistory}{%
    00 & Month Year & First issue. & \DocAuthor \\
}

\newcommand{\MetaLabelA}{Project Repository}
\newcommand{\MetaValueAUrl}{https://github.com/pwr-control/modelization-and-control.git}
\newcommand{\MetaValueALabel}{modelization-and-control.git}
\newcommand{\MetaLabelB}{Framework}
\newcommand{\MetaValueB}{MATLAB/Simulink/Simscape}

\begin{document}
\input{../../AAA_template_latex_settings/titlepage}
\tableofcontents
\listoffigures
\listoftables
\newpage
\begin{onehalfspace}

% ... body ...

\end{onehalfspace}
\end{document}
```

Documents written before `titlepage.tex` existed carry the title page inline in
their own preamble; they keep working unchanged.

## Notes

- The document class is `scrartcl`: the top sectioning level is `\section`,
  there is no `\chapter`.
- Equations are numbered per section (`\numberwithin{equation}{section}`).
- `esint` is loaded after `amsmath` on purpose: both define `\iint` and
  friends, and the four names are released before `esint` is loaded.
- `figures/` stays inside each document folder; only the logo lives here.
