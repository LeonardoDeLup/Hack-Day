"""Second Life: a local fruit assessment and recipe companion."""

from hashlib import sha256
import streamlit as st

from fruit_backend.analysis import FruitAnalyzer
from fruit_backend.errors import AnalysisError
from fruit_backend.schemas import AnalysisResult, Assessment
from recipe_catalog import Recipe, recommend_recipes
import ui_services


st.set_page_config(
    page_title="Second Life | Fruit companion",
    page_icon=":material/local_florist:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def initialize_state() -> None:
    defaults: dict[str, object] = {
        "current_upload_hash": None,
        "analysis_result": None,
        "analysis_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def get_analyzer() -> FruitAnalyzer:
    return FruitAnalyzer()


def clear_result_for_new_upload(upload_hash: str | None) -> bool:
    """Reset stale results and request analysis only when the image changes."""
    if upload_hash == st.session_state.current_upload_hash:
        return False
    st.session_state.current_upload_hash = upload_hash
    st.session_state.analysis_result = None
    st.session_state.analysis_error = None
    return upload_hash is not None


def friendly_ripeness(ripeness: str) -> str:
    return {
        "unripe": "Unripe",
        "ripe": "Ripe",
        "overripe": "Overripe",
        "uncertain": "Ripeness uncertain",
    }.get(ripeness, ripeness.replace("_", " ").title())


def friendly_spoilage(spoilage: str) -> str:
    return {
        "none_observed": "No visible spoilage observed",
        "possible": "Possible visible damage",
        "clear": "Clear visible spoilage",
        "uncertain": "Condition uncertain",
    }.get(spoilage, spoilage.replace("_", " ").title())


def badge_color(value: str, category: str) -> str:
    if category == "ripeness":
        return {"ripe": "green", "unripe": "yellow", "overripe": "orange", "uncertain": "gray"}.get(value, "gray")
    return {"none_observed": "green", "possible": "orange", "clear": "red", "uncertain": "gray"}.get(value, "gray")


def render_header() -> None:
    header_left, header_right = st.columns([4, 1], vertical_alignment="center")
    with header_left:
        st.markdown(":green-badge[Second Life]", unsafe_allow_html=False)
        st.title("Give good fruit a second life.", anchor=False)
        st.markdown("Photograph one fruit. We’ll read its ripeness, then suggest what to make next.")
    with header_right:
        st.caption(":material/lock: local model", unsafe_allow_html=False)
        st.caption("Qwen3-VL 4B Instruct")


def render_upload() -> bytes | None:
    with st.container(border=True):
        st.subheader("Start with one fruit", anchor=False)
        st.caption("A clear, well-lit photo works best. One fruit per image keeps the assessment focused.")

        source = st.segmented_control(
            "Choose a photo source",
            options=["Upload photo", "Use camera"],
            default="Upload photo",
            key="photo_source",
        )

        if source == "Use camera":
            uploaded = st.camera_input("Take a photo of your fruit", key="fruit_camera")
            empty_message = "Allow camera access, then take a picture to analyze it automatically."
        else:
            uploaded = st.file_uploader(
                "Upload a fruit photo",
                type=["jpg", "jpeg", "png", "webp"],
                max_upload_size=10,
                accept_multiple_files=False,
                key="fruit_upload",
                help="JPEG, PNG, or WebP up to 10 MiB.",
            )
            empty_message = "Drop a photo here to start analyzing automatically."

        if uploaded is None:
            st.info(empty_message, icon=":material/add_a_photo:")
            return None

        image_bytes = uploaded.getvalue()
        st.image(image_bytes, width="stretch", alt="Fruit photo selected for analysis")
        st.caption("Analyzed locally with Qwen3-VL 4B Instruct.")
        return image_bytes


def render_assessment(result: AnalysisResult | None) -> None:
    with st.container(border=True):
        st.subheader("What I see", anchor=False)
        if result is None:
            st.caption("Your assessment will appear here after analysis.")
            st.info("The image stays on this device while the local model works.", icon=":material/visibility:")
            return

        assessment = result.assessment
        if assessment.status != "assessed":
            titles = {
                "no_fruit": "Not Identified",
                "multiple_fruits": "Multiple Fruits Detected",
                "unclear_image": "Unable to Identify",
            }

            messages = {
                "no_fruit": "I couldn’t find a fruit in that image.",
                "multiple_fruits": "Please upload one fruit at a time.",
                "unclear_image": "Try uploading a clearer photograph.",
            }

            st.markdown(
                f"## {titles.get(assessment.status, 'Not Identified')}"
            )

            st.warning(
                messages.get(
                    assessment.status,
                    "I couldn’t assess this image."
                ),
                icon=":material/photo_camera:"
            )

            for observation in assessment.observations:
                st.markdown(f"- {observation}")

            st.write(assessment.explanation)
            return

        fruit_name = assessment.fruit_name or "Fruit"
        st.markdown(f"## {fruit_name.title()}")
        ripeness_col, condition_col = st.columns(2)
        with ripeness_col:
            st.caption("Ripeness")
            st.badge(friendly_ripeness(assessment.ripeness), color=badge_color(assessment.ripeness, "ripeness"))
        with condition_col:
            st.caption("Visible condition")
            st.badge(friendly_spoilage(assessment.visible_spoilage), color=badge_color(assessment.visible_spoilage, "spoilage"))

        st.write(assessment.explanation)
        with st.expander("Visible clues", expanded=True):
            for observation in assessment.observations:
                st.markdown(f"- {observation}")
        st.caption("Visual assessment only. A photo cannot confirm food safety or hidden condition.")


def render_recipe_card(recipe: Recipe) -> None:
    with st.container(border=True, height="stretch"):
        st.markdown(f"### {recipe.title}")
        st.caption(f"{recipe.prep_minutes + recipe.cook_minutes} min · serves {recipe.servings}")
        st.write(recipe.summary)
        st.markdown(f"**Why it fits**  \n{recipe.why_it_fits}")
        if recipe.tags:
            st.markdown(" ".join(f":green-badge[{tag}]" for tag in recipe.tags), unsafe_allow_html=False)
        with st.expander("Ingredients and steps"):
            st.markdown("**Ingredients**")
            for ingredient in recipe.ingredients:
                st.markdown(f"- {ingredient}")
            st.markdown("**Steps**")
            for index, step in enumerate(recipe.steps, 1):
                st.markdown(f"{index}. {step}")


def render_recipes(assessment: Assessment | None) -> None:
    st.subheader("Make something of it", anchor=False)
    if assessment is None:
        st.caption("Recipes will appear after a fruit has been assessed.")
        return
    if assessment.status != "assessed":
        st.info("A single identified fruit is needed before I can choose recipes.", icon=":material/restaurant:")
        return
    if assessment.visible_spoilage == "clear":
        st.warning("I’m holding back recipe suggestions because the image shows clear visible spoilage.", icon=":material/warning:")
        return
    if assessment.visible_spoilage in {"possible", "uncertain"}:
        st.warning("The visible condition is not fully certain. Review the fruit yourself before cooking.", icon=":material/info:")

    recipes = recommend_recipes(assessment)
    if not recipes:
        st.info(f"I don’t have a curated recipe for {assessment.fruit_name} yet.", icon=":material/menu_book:")
        return

    st.caption("A few local catalog ideas, matched to the fruit and its visible ripeness.")
    columns = st.columns(len(recipes), gap="medium")
    for column, recipe in zip(columns, recipes):
        with column:
            render_recipe_card(recipe)


def render_error(error: AnalysisError | None) -> None:
    if error is None:
        return
    messages = {
        "invalid_image": ("That image needs another look", "Upload an intact JPEG, PNG, or WebP under 10 MiB."),
        "service_unavailable": ("The local model is offline", "Start Ollama with `ollama serve`, then try again."),
        "model_missing": ("The fruit model is not installed", "Install `qwen3-vl:4b-instruct` in Ollama, then try again."),
        "timeout": ("The local model took too long", "Check system load and retry the analysis."),
        "invalid_model_output": ("The model could not finish an assessment", "Try the same photo again or upload a clearer one."),
        "inference_failed": ("The local model returned an error", "Check Ollama and retry the analysis."),
    }
    title, body = messages.get(error.code, ("The assessment did not complete", str(error)))
    st.error(body, title=title, icon=":material/error:")


initialize_state()
render_header()

left, right = st.columns([1.05, 0.95], gap="large", vertical_alignment="top")
with left:
    image_bytes = render_upload()
    upload_hash = sha256(image_bytes).hexdigest() if image_bytes is not None else None
    should_analyze = clear_result_for_new_upload(upload_hash)

with right:
    assessment_slot = st.empty()

# Streamlit reruns when a new upload is selected or a camera photo is captured.
# The hash check prevents re-inference on subsequent UI reruns.
if should_analyze and image_bytes is not None:
    with assessment_slot.container():
        process = st.status("Reading your fruit…", expanded=True)
        try:
            with process:
                st.write("Analyzing the image locally with Qwen3-VL 4B Instruct…")
                result = ui_services.analyze_uploaded_image(image_bytes, get_analyzer())
            st.session_state.analysis_result = result
            st.session_state.analysis_error = None
            process.update(label="Assessment ready", state="complete", expanded=False)
        except AnalysisError as error:
            st.session_state.analysis_result = None
            st.session_state.analysis_error = error
            process.update(label="Assessment stopped", state="error", expanded=False)

    # The upload hash is already stored, so this rerun only displays the result.
    st.rerun()

with assessment_slot.container():
    render_assessment(st.session_state.analysis_result)
    render_error(st.session_state.analysis_error)

st.space("large")
render_recipes(st.session_state.analysis_result.assessment if st.session_state.analysis_result else None)
