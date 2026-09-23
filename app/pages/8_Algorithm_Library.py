import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import FAMILY_PAGE, link, page  # noqa: E402,I001

import streamlit as st  # noqa: E402

from schedlab.algorithms import get  # noqa: E402
from schedlab.recommender import CATEGORIES, KNOWLEDGE_BASE  # noqa: E402

page("Algorithm Library", "📚", "Every algorithm in the knowledge base, with its trade-offs.")

c1, c2, c3 = st.columns([2, 2, 1])
query = c1.text_input("Search", placeholder="e.g. deadline, starvation, heterogeneous")
cats = c2.multiselect("Category", CATEGORIES)
only_impl = c3.toggle("Runnable only")

shown = 0
for info in KNOWLEDGE_BASE:
    blob = " ".join([info.name, info.category, info.summary, info.applications, *info.pros, *info.cons])
    if query and query.lower() not in blob.lower():
        continue
    if cats and info.category not in cats:
        continue
    if only_impl and not info.implemented:
        continue
    shown += 1
    badge = "▶ runnable" if info.implemented else "📖 reference only"
    with st.expander(f"**{info.name}** · {info.category} · {badge}"):
        st.write(info.summary)
        a, b = st.columns(2)
        a.markdown(f"**Time:** {info.time}  \n**Space:** {info.space}  \n"
                   f"**Parameters:** {', '.join(info.params)}")
        b.markdown(f"**Applications:** {info.applications}")
        a.markdown("**Advantages**\n" + "\n".join(f"- {p}" for p in info.pros))
        b.markdown("**Limitations**\n" + "\n".join(f"- {c}" for c in info.cons))
        if info.implemented:
            keys = info.registry_keys
            st.caption("Implementation: " + ", ".join(f"`schedlab.algorithms.REGISTRY['{k}']`" for k in keys))
            link(FAMILY_PAGE[get(keys[0]).family], f"Open {info.name} in the lab")
st.caption(f"{shown} of {len(KNOWLEDGE_BASE)} algorithms shown.")
