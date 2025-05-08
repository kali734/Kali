import json
import time
import streamlit as st

st.set_page_config(page_title="MnemonicHunter Dashboard", layout="wide")
st.title("📊 MnemonicHunter Live Dashboard")

tested_slot = st.empty()
speed_slot  = st.empty()
elapsed_slot = st.empty()
hits_slot   = st.empty()

def load_hits():
      try:
        lines = open("hits.txt","r").read().strip().splitlines()
        return lines[-10:]
      except:
        return []

while True:
                                        # metrics.json पढ़ो
    try:
        data = json.load(open("metrics.json"))
        tested_slot.metric("Phrases Tested", data["tested"])
        speed_slot.metric("Speed (p/s)", data["speed"])
        elapsed_slot.metric("Elapsed (s)", data["elapsed"])
    except:
        st.info("Waiting for metrics…")
    st.subheader("Recent Hits")
    hits = load_hits()
    if hits:
        hits_slot.write("\n".join(hits))
    else:
        hits_slot.write("No hits yet.")
    time.sleep(2)
                                                                                                        