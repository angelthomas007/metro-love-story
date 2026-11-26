"""
Metro Love Story - Streamlit app
Run:  streamlit run streamlit_app.py
Install: pip install streamlit pandas numpy matplotlib scikit-learn
"""
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

SKIN = "#ffd8b5"

# ---------- drawing helpers and the 9 scenes ----------
def person(ax, x, y, color, hair, boy=False, s=1.0, name=None):
    ax.add_patch(Circle((x, y + 12 * s), 2.4 * s, color=hair, zorder=4))
    ax.add_patch(Circle((x, y + 11 * s), 2.2 * s, color=SKIN, zorder=5))
    for dx in (-0.8, 0.8):
        ax.add_patch(Circle((x + dx * s, y + 11.3 * s), 0.25 * s, color="#222", zorder=6))
    ax.plot([x - .8 * s, x, x + .8 * s], [y + 10.2 * s, y + 9.8 * s, y + 10.2 * s],
            color="#222", lw=0.8, zorder=6)
    if boy:
        ax.add_patch(Rectangle((x - 1.8 * s, y + 4 * s), 3.6 * s, 6 * s, color=color, zorder=5))
        ax.add_patch(Rectangle((x - 1.8 * s, y + 2 * s), 3.6 * s, 2 * s, color="#1d3557", zorder=5))
    else:
        ax.add_patch(Polygon([(x - 1.5 * s, y + 9 * s), (x + 1.5 * s, y + 9 * s),
                              (x + 3 * s, y + 3 * s), (x - 3 * s, y + 3 * s)],
                             color=color, zorder=5))
    for dx in (-.8, .8):
        ax.plot([x + dx * s] * 2, [y, y + 2.5 * s], color=SKIN, lw=2, zorder=3)
    if name:
        ax.text(x, y - 2.2, name, ha="center", fontsize=8, color="#222", zorder=7)


def car(ax, x, y, color="#2a9d8f"):
    ax.add_patch(Polygon([(x + 8, y + 7), (x + 13, y + 14), (x + 29, y + 14), (x + 34, y + 7)],
                         color=color, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + 2), 42, 8, boxstyle="round,pad=0.5", color=color, zorder=3))
    for wx in (x + 9, x + 33):
        ax.add_patch(Circle((wx, y + 2), 3, color="#222", zorder=4))
        ax.add_patch(Circle((wx, y + 2), 1.2, color="#ccc", zorder=5))


def stage(title, caption, sky="#bfe6ff", night=False):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(-8, 60); ax.axis("off")
    ax.add_patch(Rectangle((0, 12), 100, 48, color="#0b1d3a" if night else sky))
    ax.add_patch(Rectangle((0, 4), 100, 8, color="#d9d4cc" if not night else "#4a4a5a"))
    ax.add_patch(Rectangle((0, -8), 100, 12, color="#5a5f66"))
    for x in range(0, 100, 14):
        ax.plot([x, x + 7], [-2, -2], color="white", lw=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.text(50, -6.2, caption, ha="center", va="center", fontsize=9, color="white", wrap=True)
    return fig, ax


def heart(ax, x, y, size=18):
    ax.text(x, y, "\u2665", color="#e63946", fontsize=size, ha="center", zorder=9)


# ---------- the scenes ----------
def scene1():
    fig, ax = stage("Scene 1: Outside the metro station",
                    "Meera and Anya step out of the metro. Riya chats with Kabir. Arjun and Dev sit on a car.")
    ax.add_patch(Rectangle((30, 12), 26, 30, color="#f1d9b5"))
    ax.add_patch(Rectangle((28, 42), 30, 6, color="#e63946"))
    ax.text(43, 43.5, "METRO", color="white", ha="center", fontweight="bold")
    ax.add_patch(Rectangle((38, 12), 10, 18, color="#4a4e69"))
    person(ax, 40, 12, "#ff7aa2", "#6b3e26", name="Meera")
    person(ax, 46, 12, "#ffd166", "#222", name="Anya")
    person(ax, 10, 6, "#9b5de5", "#c1440e", name="Riya")
    person(ax, 20, 6, "#3a86ff", "#222", boy=True, name="Kabir")
    car(ax, 62, 5)
    person(ax, 74, 14, "#f77f00", "#222", boy=True, s=0.9)
    person(ax, 84, 14, "#6a994e", "#8d5524", boy=True, s=0.9)
    ax.text(74, 31, "Arjun", ha="center", fontsize=8); ax.text(84, 31, "Dev", ha="center", fontsize=8)
    return fig


def scene2():
    fig, ax = stage("Scene 2: Everyone gets in the car and drives",
                    "Arjun drives. In the mirror he sees Meera for the very first time.")
    car(ax, 30, 5)
    ax.add_patch(Rectangle((36, 13), 24, 9, color="#cfe9ff", zorder=3))
    for x, c, h in [(40, "#f77f00", "#222"), (47, "#ff7aa2", "#6b3e26"), (54, "#ffd166", "#222")]:
        ax.add_patch(Circle((x, 17), 2.2, color=SKIN, zorder=4))
        ax.add_patch(Circle((x, 18), 2.4, color=h, zorder=3.5))
    for i in range(4):
        ax.plot([8, 24], [8 + i * 2, 8 + i * 2], color="white", lw=2, alpha=.7)
    heart(ax, 40, 27, 22)
    ax.text(40, 33, "Wow...", ha="center", fontsize=10, style="italic")
    return fig


def scene3():
    fig, ax = stage("Scene 3: Their eyes meet",
                    "Meera looks up at the mirror. Their eyes meet for one second. Both smile shyly.",
                    sky="#ffe5ec")
    for x, hair, name in [(35, "#222", "Arjun"), (65, "#6b3e26", "Meera")]:
        ax.add_patch(Circle((x, 30), 11, color=hair, zorder=2))
        ax.add_patch(Circle((x, 28), 10, color=SKIN, zorder=3))
        for dx in (-3.5, 3.5):
            ax.add_patch(Circle((x + dx, 30), 1.2, color="#222", zorder=4))
        ax.plot([x - 3, x, x + 3], [24, 22.5, 24], color="#c1440e", lw=2, zorder=4)
        ax.text(x, 12, name, ha="center", fontsize=10)
    heart(ax, 50, 36, 30)
    return fig


def scene4():
    fig, ax = stage("Scene 4: At the restaurant",
                    "All six friends share a big table, laugh and eat together.", sky="#fff1d6")
    ax.add_patch(Rectangle((20, 40), 60, 9, color="#e76f51"))
    ax.text(50, 42.5, "RESTAURANT", color="white", ha="center", fontweight="bold")
    ax.add_patch(Rectangle((22, 20), 56, 4, color="#8d5524", zorder=6))
    people = [(28, "#f77f00", "#222", True), (40, "#ff7aa2", "#6b3e26", False),
              (52, "#3a86ff", "#222", True), (64, "#ffd166", "#222", False),
              (74, "#6a994e", "#8d5524", True), (46, "#9b5de5", "#c1440e", False)]
    for x, c, h, b in people[:5]:
        person(ax, x, 12, c, h, boy=b, s=0.9)
    for x in (30, 42, 54, 66):
        ax.add_patch(Circle((x, 22), 2, color="white", ec="#999", zorder=7))
    ax.text(50, 30, "Arjun keeps glancing at Meera...", ha="center", fontsize=9, style="italic")
    return fig


def scene5():
    fig, ax = stage("Scene 5: The three girls go home together",
                    "Meera, Anya and Riya reach Anya's house and talk about the day all evening.",
                    sky="#ffb88c")
    ax.add_patch(Circle((85, 50), 5, color="#fff3b0"))
    ax.add_patch(Rectangle((40, 12), 30, 20, color="#f4a261"))
    ax.add_patch(Polygon([(37, 32), (55, 46), (73, 32)], color="#9c3d2f"))
    ax.add_patch(Rectangle((52, 12), 6, 11, color="#6d4c41"))
    person(ax, 18, 8, "#ff7aa2", "#6b3e26", name="Meera")
    person(ax, 28, 8, "#ffd166", "#222", name="Anya")
    person(ax, 36, 8, "#9b5de5", "#c1440e", name="Riya")
    ax.text(24, 30, "He looked at me!", fontsize=9, style="italic",
            bbox=dict(boxstyle="round", fc="white"))
    return fig


def scene6():
    random.seed(3)
    fig, ax = stage("Scene 6: New Year party - they meet again",
                    "At the New Year function Arjun and Meera see each other once more.",
                    night=True)
    for _ in range(9):
        x, y = random.randint(8, 92), random.randint(34, 56)
        c = random.choice(["#ffd166", "#ef476f", "#06d6a0", "#118ab2"])
        for a in range(0, 360, 45):
            import math
            ax.plot([x, x + 4 * math.cos(math.radians(a))], [y, y + 4 * math.sin(math.radians(a))],
                    color=c, lw=1.5)
    ax.text(50, 28, "HAPPY NEW YEAR", color="#ffd166", ha="center", fontsize=16, fontweight="bold")
    for x, c, h, b in [(15, "#3a86ff", "#222", True), (25, "#9b5de5", "#c1440e", False),
                       (75, "#6a994e", "#8d5524", True), (85, "#ffd166", "#222", False)]:
        person(ax, x, 8, c, h, boy=b, s=0.9)
    person(ax, 42, 8, "#f77f00", "#222", boy=True, name="Arjun")
    person(ax, 58, 8, "#ff7aa2", "#6b3e26", name="Meera")
    heart(ax, 50, 22, 22)
    return fig


def scene7():
    fig, ax = stage("Scene 7: The party is over",
                    "Everyone dances, eats cake, says goodbye and goes home. Arjun waves to Meera.",
                    night=True)
    car(ax, 55, 5, "#e63946")
    for i, x in enumerate((10, 20, 30, 40)):
        person(ax, x, 8, ["#ff7aa2", "#3a86ff", "#ffd166", "#6a994e"][i], "#222", boy=i % 2 == 1, s=0.9)
    ax.text(25, 28, "Bye! Happy New Year!", color="white", ha="center", fontsize=10)
    return fig


def scene8():
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.set_xlim(0, 50); ax.set_ylim(0, 80); ax.axis("off")
    ax.add_patch(FancyBboxPatch((5, 3), 40, 74, boxstyle="round,pad=1", fc="#fafafa", ec="#222", lw=3))
    ax.add_patch(Circle((15, 68), 4.5, fc="#ff7aa2", ec="#c13584", lw=3))
    ax.text(22, 67, "meera_smiles", fontsize=10, fontweight="bold")
    ax.add_patch(Rectangle((10, 40), 30, 20, color="#ffe5ec"))
    ax.text(25, 50, "New Year photo", ha="center", fontsize=10)
    msgs = [(12, 34, "arjun_vibes: Nice pic!", "#e0f0ff"),
            (14, 27, "meera_smiles: Thanks :)", "#ffe0ec"),
            (12, 20, "arjun_vibes: Remember me?", "#e0f0ff"),
            (14, 13, "meera_smiles: The metro boy!", "#ffe0ec")]
    for x, y, t, c in msgs:
        ax.text(x, y, t, fontsize=9, bbox=dict(boxstyle="round", fc=c, ec="none"))
    ax.set_title("Scene 8: Instagram story chat", fontweight="bold")
    return fig


def scene9():
    fig, ax = stage("Scene 9: Love",
                    "Chat turned into calls, calls into dates. They fell in love. THE END.",
                    sky="#ffd6e0")
    person(ax, 42, 10, "#f77f00", "#222", boy=True, s=1.3, name="Arjun")
    person(ax, 58, 10, "#ff7aa2", "#6b3e26", s=1.3, name="Meera")
    for x, y, sz in [(50, 38, 40), (30, 30, 20), (70, 30, 20), (20, 44, 14), (80, 44, 14)]:
        heart(ax, x, y, sz)
    return fig




STORY = [
    (scene1, "Meera and Anya step out of the metro station. Riya and Kabir stand outside talking, and Arjun and Dev sit on a car."),
    (scene2, "Everyone gets into the car and Arjun drives. In the mirror he sees Meera for the very first time."),
    (scene3, "Meera looks up and their eyes meet for one second. Both smile shyly."),
    (scene4, "All six friends reach the restaurant, share a big table and laugh together. Arjun keeps glancing at Meera."),
    (scene5, "The three girls, Meera, Anya and Riya, go to Anya's house and talk about the day all evening."),
    (scene6, "At the New Year function, Arjun and Meera see each other once more."),
    (scene7, "The party is over. Everyone says goodbye and goes home. Arjun waves to Meera."),
    (scene8, "On Instagram, Arjun replies to Meera's New Year story. The chat begins."),
    (scene9, "The chats become calls, the calls become dates. They fall in love. THE END."),
]


@st.cache_resource
def train_models():
    """Make practice data and train 3 models (runs only once)."""
    rng = np.random.default_rng(42)
    n = 300
    data = pd.DataFrame({
        "eye_contact": rng.integers(0, 2, n),
        "met_again": rng.integers(0, 2, n),
        "story_replies": rng.integers(0, 2, n),
        "chat_days": rng.integers(0, 60, n),
        "common_interests": rng.integers(0, 11, n),
    })
    score = (data["eye_contact"] * 2 + data["met_again"] * 2 + data["story_replies"] * 3
             + data["chat_days"] / 10 + data["common_interests"] / 2
             + rng.normal(0, 1.5, n))
    data["love"] = (score > 8).astype(int)
    X, y = data.drop("love", axis=1), data["love"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }
    accuracy = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy[name] = accuracy_score(y_test, model.predict(X_test))
    return models, accuracy


def go_next():
    st.session_state.scene = min(st.session_state.scene + 1, len(STORY) - 1)


def go_back():
    st.session_state.scene = max(st.session_state.scene - 1, 0)


def restart():
    st.session_state.scene = 0


def story_tab():
    if "scene" not in st.session_state:
        st.session_state.scene = 0
    i = st.session_state.scene
    scene_fn, text = STORY[i]

    st.progress((i + 1) / len(STORY), text=f"Scene {i + 1} of {len(STORY)}")
    fig = scene_fn()
    st.pyplot(fig)
    plt.close(fig)
    st.write(text)

    c1, c2, c3 = st.columns(3)
    c1.button("Previous", on_click=go_back, disabled=(i == 0), use_container_width=True)
    c2.button("Next", on_click=go_next, disabled=(i == len(STORY) - 1), use_container_width=True)
    c3.button("Restart", on_click=restart, use_container_width=True)


def ml_tab():
    st.subheader("Will they fall in love?")
    st.write("Change the story details and let 3 machine learning models guess.")
    models, accuracy = train_models()

    eye = st.checkbox("They looked at each other in the car", value=True)
    again = st.checkbox("They met again at the New Year function", value=True)
    reply = st.checkbox("She replied to his Instagram story", value=True)
    days = st.slider("Days they chatted", 0, 59, 30)
    interests = st.slider("Common interests (0 to 10)", 0, 10, 7)

    pair = pd.DataFrame([{
        "eye_contact": int(eye), "met_again": int(again), "story_replies": int(reply),
        "chat_days": days, "common_interests": interests,
    }])

    st.markdown("**Predictions**")
    for name, model in models.items():
        chance = model.predict_proba(pair)[0][1]
        verdict = "Love" if chance >= 0.5 else "Just friends"
        st.write(f"{name}: {verdict} ({chance:.0%} chance of love)")
        st.progress(float(chance))

    st.markdown("**How accurate is each model?**")
    st.bar_chart(pd.Series(accuracy, name="accuracy"))


def main():
    st.set_page_config(page_title="Metro Love Story", layout="centered")
    st.title("Metro Love Story")
    tab1, tab2 = st.tabs(["The story", "Machine learning"])
    with tab1:
        story_tab()
    with tab2:
        ml_tab()


if __name__ == "__main__":
    main()
