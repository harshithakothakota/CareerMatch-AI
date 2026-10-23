import json
import re
import time

import streamlit as st
from langchain_ollama import OllamaLLM


@st.cache_resource
def get_llm():
    """Reuse the local Ollama model across Streamlit reruns."""
    return OllamaLLM(
        model="llama3.2:1b",
        num_predict=1024,
        temperature=0,
        num_ctx=4096,
    )


def _source_label(document):
    """Build a readable source label from document metadata."""
    metadata = getattr(document, "metadata", None) or {}
    filename = (
        metadata.get("source_filename")
        or metadata.get("source")
        or "Uploaded document"
    )
    page = metadata.get("page_number")
    if page is None and metadata.get("page") is not None:
        page = metadata["page"] + 1
    page_text = f", page {page}" if page is not None else ""
    document_type = metadata.get("document_type", "Document")
    return f"{filename}{page_text} | {document_type}"


def _format_context(documents):
    """Convert document chunks into readable source-labelled text."""
    sections = []
    for document in documents or []:
        content = getattr(document, "page_content", "")
        if content:
            sections.append(f"[SOURCE: {_source_label(document)}]\n{content}")
    return "\n\n".join(sections) or "No readable document content was retrieved."


def retrieve_documents(vector_store, query):
    """Retrieve relevant chunks from both uploaded documents."""
    resume_results = vector_store.similarity_search(
        query, k=2, filter={"document_type": "Resume"}
    )
    jd_results = vector_store.similarity_search(
        query, k=2, filter={"document_type": "Job Description"}
    )
    return resume_results + jd_results


def _document_text(documents, expected_type):
    """Join content only from documents of the requested metadata type."""
    selected = _filter_by_type(documents or [], expected_type)
    return "\n".join(
        str(getattr(doc, "page_content", "") or "") for doc in selected
    ).strip()


def _extract_required_skills(jd_text):
    """Extract a labelled required-skills list from common JD layouts."""
    if not jd_text:
        return []

    # Prefer the explicit skills section, bounded by the next common section.
    match = re.search(
        r"(?:required\s+skills|skills\s+required|essential\s+skills)\s*:\s*(.*?)(?=\b(?:responsibilities|preferred\s+qualifications|minimum\s+qualifications|requirements|role\s+overview|qualifications)\s*:|$)",
        jd_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        # Handle headings where requirements are on subsequent lines.
        match = re.search(
            r"(?:required\s+skills|skills\s+required|essential\s+skills)\s*\n(.*?)(?=\n\s*(?:responsibilities|preferred\s+qualifications|minimum\s+qualifications|requirements|role\s+overview|qualifications)\s*:|$)",
            jd_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    if not match:
        return []

    section = match.group(1)
    # PDF extractors can represent bullets as unusual control-like characters.
    section = section.replace("\x7f", "\n").replace("•", "\n").replace("▪", "\n")
    section = re.sub(r"(?m)^\s*[-*]+\s*", "", section)
    pieces = re.split(r"[\n,;|]+", section)
    items = []
    for piece in pieces:
        item = re.sub(r"^\s*\d+[.)]\s*", "", piece).strip(" \t\r\n:.-")
        if not item:
            continue
        # When a PDF collapses all bullet items into one line, split on the
        # bullet remnants or retain a phrase only if it looks like one item.
        if len(item) > 80:
            continue
        if item.casefold() not in {x.casefold() for x in items}:
            items.append(item)
    return items


def _extract_resume_skill_section(resume_text):
    """Return text explicitly following a resume skills heading, if present."""
    match = re.search(
        r"(?:technical\s+skills|key\s+skills|core\s+skills|skills|technologies|tools)\s*:\s*(.*?)(?=\n\s*(?:projects?|experience|education|certifications?|summary|profile)\s*:|$)",
        resume_text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _phrase_present(phrase, text):
    """Case-insensitive whole-phrase check, tolerant of punctuation/spacing."""
    phrase_words = re.findall(r"[a-z0-9+#.]+", str(phrase).casefold())
    text_words = re.findall(r"[a-z0-9+#.]+", str(text).casefold())
    # Normalize simple plurals (for example, API/APIs) so equivalent wording
    # in a resume and JD does not create a false gap.
    normalize = lambda word: word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word
    phrase_words = [normalize(word) for word in phrase_words]
    text_words = [normalize(word) for word in text_words]
    if not phrase_words or not text_words or len(phrase_words) > len(text_words):
        return False
    n = len(phrase_words)
    return any(text_words[i:i+n] == phrase_words for i in range(len(text_words)-n+1))


def _deterministic_skill_comparison(query, documents):
    """Answer straightforward cross-document skill comparisons from source text."""
    q = (query or "").casefold()
    comparison_intent = any(term in q for term in (
        "compare", "comparison", "match", "matching", "overlap", "both documents",
        "both the resume", "resume with the job", "resume against the job",
        "required skills", "skills in the resume",
    ))
    if not comparison_intent:
        return None

    resume_text = _document_text(documents, "Resume")
    jd_text = _document_text(documents, "Job Description")
    if not resume_text or not jd_text:
        return None

    required = _extract_required_skills(jd_text)
    if not required:
        return None

    # Classify each required skill using resume-only evidence.
    skills_section = _extract_resume_skill_section(resume_text)
    demonstrated, listed_only, not_mentioned = [], [], []
    for skill in required:
        status = _resume_skill_status(skill, resume_text, skills_section)
        if status == "Demonstrated":
            demonstrated.append(skill)
        elif status == "Listed":
            listed_only.append(skill)
        else:
            not_mentioned.append(skill)

    # Find skills explicitly listed in the resume but not in the JD required list.
    extra_resume_skills = []
    if skills_section and re.search(r"[,;|\n]", skills_section):
        parts = [x.strip(" \t\r\n:.-") for x in re.split(r"[,;|\n]+", skills_section)]
        for item in parts:
            if not item or len(item) > 80:
                continue
            if not any(_phrase_present(item, [s]) for s in required):
                if item.casefold() not in {x.casefold() for x in extra_resume_skills}:
                    extra_resume_skills.append(item)

    lines = ["Based on the uploaded documents:", "", "**Required skills demonstrated through project or experience evidence:**"]
    lines.extend(f"- {skill}" for skill in demonstrated) if demonstrated else lines.append("- None clearly demonstrated in the retrieved resume text.")

    lines.extend(["", "**Required skills listed in the resume, but without clear project/experience evidence:**"])
    lines.extend(f"- {skill}" for skill in listed_only) if listed_only else lines.append("- None identified.")

    lines.extend(["", "**Required skills not mentioned in the retrieved resume text:**"])
    lines.extend(f"- {skill}" for skill in not_mentioned) if not_mentioned else lines.append("- None identified from the extracted required-skills list.")

    lines.extend(["", "**Skills listed in the resume but not in the job's required-skills list:**"])
    lines.extend(f"- {skill}" for skill in extra_resume_skills) if extra_resume_skills else lines.append("- None identified.")

    lines.extend([
        "",
        "**Important distinction:** A skill listed in the resume is not automatically proof of hands-on experience. Classification as not mentioned is a documentation gap, not proof that the candidate lacks that ability.",
    ])
    return "\n".join(lines)

def generate_answer(query, documents):
    """Answer a question using retrieved resume and job-description evidence."""
    deterministic_comparison = _deterministic_skill_comparison(query, documents)
    if deterministic_comparison is not None:
        return deterministic_comparison

    context = _format_context(documents)
    prompt = f"""
You are CareerMatch AI, a resume and job description assistant.
Answer the user's question using only the document context.

Rules:
- Be accurate, clear, and concise.
- Do not invent skills, qualifications, experience, or achievements.
- Say when the documents do not provide enough information.
- Distinguish resume evidence from job-description requirements.
- Cite the exact [SOURCE: ...] label when useful.
- Do not claim a requirement is demonstrated unless the resume supports it.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

Provide a focused answer:
"""
    start = time.time()
    answer = get_llm().invoke(prompt)
    print(f"LLM answer generation took {time.time() - start:.2f} seconds")
    return str(answer).strip()


def clean_skill_gap_answer(answer):
    """Remove internal checklist text from a displayed report."""
    marker = "**FINAL CHECKS**"
    if marker in answer:
        answer = answer.split(marker)[0]
    return answer.strip()


def _metadata_type(document):
    metadata = getattr(document, "metadata", None) or {}
    return str(metadata.get("document_type", "")).strip().lower()


def _filter_by_type(documents, expected_type):
    """Keep only chunks whose metadata identifies the expected document type."""
    expected = expected_type.lower()
    return [
        doc for doc in documents
        if _metadata_type(doc) == expected
    ]


def _extract_labelled_items(text, labels):
    """
    Extract comma/semicolon/pipe-separated items following labels such as
    'Skills:' or 'Required Skills:'. This is domain-independent.
    """
    found = []
    for line in text.splitlines():
        clean = re.sub(r"^\s*[-*•#]+\s*", "", line).strip()
        for label in labels:
            match = re.match(
                rf"^{re.escape(label)}\s*:\s*(.*)$",
                clean,
                flags=re.IGNORECASE,
            )
            if match:
                value = match.group(1).strip()
                for part in re.split(r"[,;|]", value):
                    item = part.strip(" .:-\t")
                    if item and len(item) <= 100:
                        found.append(item)
                break
    # Deduplicate without losing original capitalization/order.
    unique = []
    seen = set()
    for item in found:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _render_items(items, empty_message):
    if not items:
        return [empty_message]
    return [f"- {item}" for item in items]


def _extract_jd_section_items(jd_text, section_patterns, stop_patterns):
    """Extract complete bullet/paragraph criteria from a JD section.

    Handles PDF text where a heading is inline, criteria wrap across lines, and
    the phrase "Experience with ..." follows a degree requirement on the same
    line. It intentionally does not treat document titles or company metadata
    as qualifications.
    """
    text = (jd_text or "").replace("\r", "\n")
    text = text.replace("\x7f", "\n").replace("•", "\n").replace("▪", "\n")

    # Give recognized headings their own line even if PDF extraction joined them
    # to the previous sentence. Headings may or may not have a colon.
    all_headings = section_patterns + stop_patterns + [
        r"company", r"job\s*title", r"position", r"location",
        r"employment\s*type", r"role\s*overview",
    ]
    for heading in all_headings:
        text = re.sub(
            rf"(?i)(?<!^)(?<!\n)\s+(?=(?:{heading})\s*:?)",
            "\n",
            text,
        )

    # Drop title-only lines before looking for the target section.
    lines = []
    for line in text.splitlines():
        item = line.strip()
        if re.fullmatch(r"(?i)\s*[A-Z][A-Z\s&/–—-]{2,}\s*[-–—:]\s*JOB\s+(?:DESCRIPTION|POSTING|ADVERTISEMENT)\s*", item):
            continue
        if re.fullmatch(r"(?i)\s*(?:JOB\s+DESCRIPTION|JOB\s+POSTING|JOB\s+ADVERTISEMENT)\s*", item):
            continue
        lines.append(line)
    text = "\n".join(lines)

    section_re = re.compile(
        r"(?im)^\s*(?:" + "|".join(section_patterns) + r")\s*:\s*"
    )
    match = section_re.search(text)
    if not match:
        return []

    body = text[match.end():]
    stop_re = re.compile(
        r"(?im)^\s*(?:" + "|".join(stop_patterns + [
            r"company", r"job\s*title", r"position", r"location",
            r"employment\s*type", r"role\s*overview",
        ]) + r")\s*:\s*"
    )
    stop = stop_re.search(body)
    if stop:
        body = body[:stop.start()]

    def is_noise(value):
        value = value.strip()
        return (
            not value
            or bool(re.search(r"\b(?:JOB\s+DESCRIPTION|JOB\s+POSTING|JOB\s+ADVERTISEMENT)\b", value, re.I))
            or bool(re.match(r"^(?:company|job\s*title|position|location|employment\s*type|note|disclaimer)\s*:", value, re.I))
            or bool(re.match(r"^(?:this is (?:a )?fictional|fictional test data|for testing purposes only|example only)\b", value, re.I))
            or bool(re.fullmatch(r"(?:preferred qualifications|preferred skills|nice to have|desired qualifications|bonus qualifications)", value, re.I))
        )

    raw_items = []
    for line in body.splitlines():
        item = re.sub(r"^\s*(?:[-*•▪]+|\d+[.)])\s*", "", line).strip()
        if not item or is_noise(item):
            continue
        # Keep a degree as one item, but split a second sentence that starts
        # with "Experience with ..." into a separate, complete qualification.
        parts = re.split(r"(?<=[.!?])\s+(?=Experience\b)", item, flags=re.I)
        for part in parts:
            part = part.strip(" \t\r\n:.-")
            if not part or is_noise(part):
                continue
            # Join PDF-wrapped continuation lines to the preceding criterion.
            if raw_items and raw_items[-1] == "__JOIN_EXPERIENCE__":
                raw_items[-1] = "Experience " + part
            elif raw_items and re.match(r"^(?:with|and|or|of|for|to|in|on|using|including|such as|as well as)\b", part, re.I):
                raw_items[-1] = raw_items[-1].rstrip(" .;:-") + " " + part
            elif re.fullmatch(r"experience", part, re.I):
                # Wait for its continuation instead of emitting a dangling item.
                raw_items.append("__JOIN_EXPERIENCE__")
            else:
                raw_items.append(part)

    items, seen = [], set()
    for raw in raw_items:
        if raw == "__JOIN_EXPERIENCE__":
            continue
        item = re.sub(r"\s+", " ", raw).strip(" \t\r\n:.-")
        if not item or len(item) > 220 or is_noise(item):
            continue
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            items.append(item)
    return items

def _extract_preferred_qualifications(jd_text):
    """Extract preferred/nice-to-have criteria from common JD headings."""
    return _extract_jd_section_items(
        jd_text,
        [r"preferred qualifications", r"preferred skills", r"nice to have", r"desired qualifications", r"bonus qualifications"],
        [r"responsibilities", r"required skills", r"skills required", r"essential skills", r"requirements", r"minimum qualifications", r"role overview", r"about the role", r"what you will do", r"benefits", r"about us"],
    )


def _resume_skill_status(skill, resume_text, skills_section):
    """Return Listed/Demonstrated/None using resume-only evidence."""
    if not _phrase_present(skill, resume_text):
        return None
    # Treat mention in project/experience descriptions as stronger evidence than
    # a skills-list mention. This remains generic across industries.
    evidence_sections = re.findall(
        r"(?is)(?:projects?|experience|work history|employment|responsibilities|achievements)\s*:?\s*(.*?)(?=\n\s*(?:education|certifications?|technical skills|key skills|core skills|skills|tools|technologies|summary|profile)\s*:|$)",
        resume_text,
    )
    if any(_phrase_present(skill, section) for section in evidence_sections):
        return "Demonstrated"
    if _phrase_present(skill, skills_section) or _phrase_present(skill, resume_text):
        return "Listed"
    return None


def _get_ordered_document_text(vector_store, expected_type, fallback_query, k=8):
    """Prefer all stored chunks in source order; fall back to semantic retrieval.

    Similarity search can return JD chunks out of order, which can make a section
    parser mistake a title or a continuation line for a qualification.
    """
    try:
        payload = vector_store.get(
            where={"document_type": expected_type},
            include=["documents", "metadatas"],
        )
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        rows = []
        for index, content in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            rows.append((metadata, str(content or "")))
        rows.sort(key=lambda row: (
            str(row[0].get("source_filename") or row[0].get("source") or ""),
            int(row[0].get("page", row[0].get("page_number", 0)) or 0),
            int(row[0].get("start_index", 0) or 0),
        ))
        if rows:
            # Remove exact duplicate chunks and trim overlap duplicates while
            # retaining source order. The parser also deduplicates criteria.
            unique = []
            seen = set()
            for _, content in rows:
                content = content.strip()
                if content and content not in seen:
                    seen.add(content)
                    unique.append(content)
            if unique:
                return "\n".join(unique), _filter_by_type(
                    vector_store.similarity_search(fallback_query, k=k, filter={"document_type": expected_type}),
                    expected_type,
                )
    except Exception as exc:
        print(f"Ordered document retrieval fallback for {expected_type}: {exc}")

    results = _filter_by_type(
        vector_store.similarity_search(fallback_query, k=k, filter={"document_type": expected_type}),
        expected_type,
    )
    return "\n".join(getattr(doc, "page_content", "") or "" for doc in results), results


def generate_skill_gap_analysis(vector_store):
    """Compare any resume/JD pair, with deterministic required/preferred separation."""
    start = time.time()
    try:
        resume_text, resume_results = _get_ordered_document_text(
            vector_store, "Resume",
            "resume candidate skills experience projects education tools certifications achievements",
            k=8,
        )
        jd_text, jd_results = _get_ordered_document_text(
            vector_store, "Job Description",
            "job description role responsibilities required skills preferred qualifications",
            k=8,
        )
        if not resume_results:
            return "## Resume Not Found\n\nNo chunks tagged as `Resume` were retrieved. Check the document_type metadata assigned during upload."
        if not jd_results:
            return "## Job Description Not Found\n\nNo chunks tagged as `Job Description` were retrieved. Check the document_type metadata assigned during upload."
        resume_context, jd_context = _format_context(resume_results), _format_context(jd_results)
        skills_section = _extract_resume_skill_section(resume_text)
        required = _extract_required_skills(jd_text)
        preferred = _extract_preferred_qualifications(jd_text)

        # Ask the model for context-sensitive evidence and recommendations, but
        # enforce requirement categories and skill statuses below from source text.
        prompt = f"""
You are CareerMatch AI. Analyze the resume against the job description for ANY profession.

RESUME EVIDENCE (candidate facts only):
{resume_context}

JOB DESCRIPTION (requirements only):
{jd_context}

Return one concise JSON object with keys: overall_assessment, next_steps.
- overall_assessment: evidence-based summary, no unsupported claims.
- next_steps: exactly 3 useful actions. Suggestions may recommend practice or learning,
  but must not imply the candidate already performed work not stated in the resume.
- Never treat job-description text as evidence about the candidate.
- Missing resume evidence means "not evidenced/documented", not proof of inability.
- Keep recommendations relevant to the actual role and documents.
- Return JSON only, no Markdown fences.
"""
        assessment = "The comparison is based on requirements and evidence found in the retrieved documents. Missing documentation does not prove lack of ability."
        steps = []
        try:
            response = str(get_llm().invoke(prompt)).strip()
            response = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", response, flags=re.IGNORECASE).strip()
            first, last = response.find("{"), response.rfind("}")
            if first >= 0 and last > first:
                report = json.loads(response[first:last + 1])
                if isinstance(report, dict):
                    # Keep the assessment deterministic: an LLM must not infer
                    # that a candidate lacks a degree or skill from missing text.
                    raw_steps = report.get("next_steps", [])
                    if isinstance(raw_steps, list):
                        steps = [x.strip() for x in raw_steps if isinstance(x, str) and x.strip()][:3]
        except Exception as exc:
            print(f"Skill gap narrative generation fallback: {exc}")

        lines = ["## Matching Skills"]
        matched = []
        for skill in required:
            status = _resume_skill_status(skill, resume_text, skills_section)
            if status:
                evidence = f"The resume describes this in a project/experience section." if status == "Demonstrated" else "The skill is mentioned in the resume, but hands-on evidence is not clearly described."
                matched.append((skill, status, evidence))
        if matched:
            for skill, status, evidence in matched:
                lines.append(f"- **{skill}** ({status}): {evidence}")
        else:
            lines.append("- No required skills were confirmed in the retrieved resume text.")

        matched_names = {x[0].casefold() for x in matched}
        gaps = [skill for skill in required if skill.casefold() not in matched_names]
        lines.append("\n## Potential Gaps")
        if gaps:
            for skill in gaps:
                lines.append(f"- **{skill}:** This required item was not found in the retrieved resume text. This is a documentation gap, not proof that the candidate lacks the ability.")
        else:
            lines.append("- No required-skill documentation gaps were identified from the extracted required-skills list.")

        lines.append("\n## Preferred Qualifications")
        if preferred:
            for qualification in preferred:
                if _phrase_present(qualification, resume_text):
                    status = "Evidence found"
                    reason = "The qualification is mentioned in the resume; review the surrounding context to confirm the level of experience."
                else:
                    status = "Not evidenced"
                    reason = "This preferred qualification is not explicitly documented in the retrieved resume text."
                lines.append(f"- **{qualification}** ({status}): {reason}")
        else:
            lines.append("- No preferred qualifications were identified under a recognized preferred-qualification heading in the retrieved job description.")

        lines.append("\n## Overall Assessment")
        documented_count = len(matched)
        required_count = len(required)
        if required_count:
            lines.append(
                f"The retrieved resume text documents or lists {documented_count} of "
                f"{required_count} extracted required skills. Missing documentation "
                "does not establish that the candidate lacks a skill or qualification."
            )
        else:
            lines.append(
                "No clearly labelled required-skills list could be extracted from the "
                "retrieved job description. Review the source document's formatting. "
                "No conclusion about the candidate's qualifications can be made from "
                "missing extracted requirements."
            )
        lines.append("\n## Recommended Next Steps")
        if steps:
            for step in steps:
                lines.append(f"- {step}")
        else:
            fallback_steps = []
            if gaps:
                fallback_steps.append("Add truthful project, coursework, or work examples for relevant required skills you have actually used; otherwise plan a practice exercise.")
            fallback_steps.append("Clarify the tools, methods, and outcomes in the projects already described in the resume.")
            fallback_steps.append("Choose one relevant requirement or preferred qualification to practise and document with a small, verifiable project.")
            for step in fallback_steps[:3]:
                lines.append(f"- {step}")

        print(f"Skill gap analysis completed in {time.time() - start:.2f} seconds.")
        return "\n".join(lines)
    except Exception as exc:
        print(f"Skill gap analysis error: {exc}")
        return "## Skill Gap Analysis Error\n\nThe analysis could not be completed. Check the VS Code terminal for details: " + str(exc)


def generate_interview_questions(vector_store):
    """Generate eight role-specific questions without assuming undocumented experience."""
    resume_results = _filter_by_type(vector_store.similarity_search(
        "skills projects experience education technologies achievements",
        k=5, filter={"document_type": "Resume"},
    ), "Resume")
    jd_results = _filter_by_type(vector_store.similarity_search(
        "job responsibilities required skills technologies qualifications",
        k=5, filter={"document_type": "Job Description"},
    ), "Job Description")
    if not resume_results or not jd_results:
        missing = "resume" if not resume_results else "job description"
        return f"Unable to generate interview questions: no {missing} chunks were retrieved. Check the uploaded documents and their metadata."

    resume_text = "\n".join(getattr(doc, "page_content", "") or "" for doc in resume_results)
    prompt = f"""
You are CareerMatch AI, an interview coach for ANY profession.

RESUME EVIDENCE:
{_format_context(resume_results)}

JOB DESCRIPTION EVIDENCE:
{_format_context(jd_results)}

Generate exactly 8 distinct interview questions, numbered continuously 1 through 8:
1-3: technical/domain-specific skills appropriate to this role (if the role is not technical,
     ask role-specific knowledge/process questions instead of forcing coding questions).
4-5: projects or experience actually documented in the resume.
6-7: responsibilities or requirements in the job description.
8: one behavioral question.

Rules:
- Adapt to the actual job title, industry, requirements, and resume; do not assume a particular domain.
- Never presuppose the candidate has used a tool or performed a task unless the resume documents it.
- If a relevant requirement is not evidenced in the resume, phrase the question hypothetically, e.g. "How would you approach..." or ask how they would learn it.
- Ask about documented projects as described; do not invent challenges, outcomes, deployments, team collaboration, or work history.
- Avoid duplicates, avoid generating answers, and keep each question clear.
- Use continuous numbering exactly 1. through 8.; do not restart numbering in sections.
- Return only the 8 numbered questions, optionally with short category headings.
"""
    start = time.time()
    answer = str(get_llm().invoke(prompt)).strip()
    # Normalize common model output that restarts question numbering by category.
    # Preserve question text and make the sequence easy to reference.
    raw_lines = answer.splitlines()
    question_lines = []
    for line in raw_lines:
        clean = line.strip()
        if re.match(r"^(?:[-*]\s*)?(?:\d{1,2}[.)]|Q\s*\d+[:.)])\s+\S", clean, flags=re.IGNORECASE):
            question = re.sub(r"^(?:[-*]\s*)?(?:\d{1,2}[.)]|Q\s*\d+[:.)])\s*", "", clean, flags=re.IGNORECASE).strip()
            if question and question not in question_lines:
                question_lines.append(question)
    if len(question_lines) >= 8:
        question_lines = question_lines[:8]
        # If a question presumes experience with a preferred qualification that
        # the resume does not document, rewrite it as a hypothetical prompt.
        jd_text = "\n".join(getattr(doc, "page_content", "") or "" for doc in jd_results)
        jd_terms = _extract_required_skills(jd_text) + _extract_preferred_qualifications(jd_text)
        for i, question in enumerate(question_lines):
            for term in jd_terms:
                # Only rewrite if the question asks about using/experience with a
                # criterion that the resume does not document.
                assumes_experience = bool(re.search(
                    r"\b(?:your experience with|experience using|how did you use|when you used|how you used|you used)\b",
                    question,
                    flags=re.IGNORECASE,
                ))
                if assumes_experience and _phrase_present(term, question) and not _phrase_present(term, resume_text):
                    subject = re.sub(
                        r"^(?:experience with|familiarity with|knowledge of|proficiency in|experience in)\s+",
                        "",
                        term,
                        flags=re.IGNORECASE,
                    )
                    question_lines[i] = (
                        f"How would you approach learning or using {subject} in a project relevant to this role?"
                    )
                    break
        answer = "\n".join(f"{i}. {q}" for i, q in enumerate(question_lines, 1))
    print(f"Interview question generation took {time.time() - start:.2f} seconds.")
    return answer

