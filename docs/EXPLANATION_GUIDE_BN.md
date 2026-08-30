# এই থিসিসে আমরা কী করেছি — সহজ ভাষায় গাইড

**কাদের জন্য:** টিমের যে কেউ, যে এখনো পেপারটা পড়েনি। পাঁচ মিনিটের ভার্সন চাইলে পার্ট ১ পড়ো,
পুরো কাজটা বুঝতে চাইলে পার্ট ২–১০, আর স্যারের সামনে বসার আগে অবশ্যই পার্ট ১২।

**এখানে নতুন কোনো তথ্য নেই।** পেপারে যা আছে, সেটাই সহজ ভাষায় লেখা। এই গাইডের প্রতিটা সংখ্যা
`tables/findings.tex` থেকে এসেছে, যেটা কোড দিয়ে `results/` থেকে অটোমেটিক তৈরি হয়। কোনোদিন যদি
এই ফাইলের কোনো সংখ্যা আর পেপারের সংখ্যা আলাদা হয়, তাহলে পেপারটাই ঠিক — এই ফাইলটা পুরোনো।

> **ইংরেজি ভার্সন:** `docs/EXPLANATION_GUIDE.md` — দুটোতে একই কথা আছে।

**সূচিপত্র**

| | | | |
|---|---|---|---|
| ১. এক নজরে পুরো কাজটা | ৫. এক্সপেরিমেন্টের নকশা | ৯. Pre-registration ও reproducibility | ১৩. কে কোন অংশের উত্তর দেবে |
| ২. দরকারি শব্দগুলো | ৬. ছয়টা প্রশ্ন | ১০. সীমাবদ্ধতা | ১৪. মুখস্থ রাখার সংখ্যা |
| ৩. কাজটা কেন দরকার ছিল | ৭. আমরা যা পেয়েছি | ১১. সেকশন-ভিত্তিক মানচিত্র | ১৫. শব্দকোষ |
| ৪. আমাদের hybrid method | ৮. চারটা ফাঁদ | ১২. স্যারের প্রশ্ন ও উত্তর | |


---

## ১. এক নজরে পুরো কাজটা

### ১.১ এক প্যারাগ্রাফে

কোনো machine learning model যখন একজন মানুষ সম্পর্কে সিদ্ধান্ত দেয় — ধরো, "এই রোগীর ডায়াবেটিস
হওয়ার সম্ভাবনা বেশি" — তখন model নিজে বলতে পারে না **কেন** সে এই সিদ্ধান্তে এল। এই "কেন"-টা
পরে ব্যাখ্যা করার জন্য দুটো টুল সবচেয়ে বেশি ব্যবহৃত হয়: **SHAP** আর **LIME**। এদের গণিতের ভিত্তি
সম্পূর্ণ আলাদা, তাই একই prediction-এর ওপর দুটো চালালে প্রায়ই দুটো আলাদা উত্তর আসে — কোন
feature গুরুত্বপূর্ণ, সেই ব্যাপারে দুজন একমত হয় না। কোনটাকে বিশ্বাস করা উচিত, তার কোনো
নীতিনির্ভর উত্তর এখন পর্যন্ত নেই। আমাদের থিসিস এই সমস্যায় দুটো কাজ করেছে। **প্রথমত**, দুটোকে
ঠিকভাবে তুলনা করেছি — একই data, একই model, একই metric-এর সংজ্ঞা, একই statistical treatment —
৬টা classifier × ৩টা public medical dataset = **১৮টা combination**, আর **১,৭৪৬টা আলাদা
prediction** ব্যাখ্যা করে। **দ্বিতীয়ত**, দুটোকে **একসাথে মেশানোর** একটা method প্রস্তাব করেছি:
**fidelity-weighted rank fusion** — যে explainer আলাদা রাখা data-তে বেশি faithful প্রমাণিত হয়,
তাকে বেশি weight দেয়। তুলনাটা স্পষ্ট উত্তর দিয়েছে। মেশানোর পদ্ধতিটা কাজ করেনি — এবং **কেন
করেনি সেটা আমরা ব্যাখ্যা করতে পারি**, যা নিজেই একটা ফলাফল।

### ১.২ তিনটা contribution (এগুলো মুখস্থ রাখো)

১. **একটা reproducible multi-axis benchmark।** দুটো explainer, তিনটা fusion rule, একটা
   random-ranking control — ছয়টা classifier আর তিনটা dataset জুড়ে, একটাই leakage-safe
   protocol-এ, এবং **যেকোনো ফলাফল বের হওয়ার আগেই** লিখে রাখা এগারোটা hypothesis নিয়ে।
২. **Explanation stability-কে তিন ভাগে ভাগ করা।** Explainer-এর randomness, input-এ ছোট
   পরিবর্তন, আর model retrain করার variance — এই তিনটা আলাদা জিনিস। যারা একটামাত্র
   "stability" সংখ্যা রিপোর্ট করে, তারা এই তিনটাকে গুলিয়ে ফেলে — আর এতে conclusion বদলে যায়।
৩. **একটা hybrid consensus method, এবং তার সীমা নিয়ে সৎ হিসাব।** Fidelity-weighted rank
   fusion আমাদের; এটাকে দুটো published fusion baseline আর একটা কঠিন মানদণ্ডের বিরুদ্ধে পরীক্ষা
   করা হয়েছে। সেই মানদণ্ড এটা পার করতে পারেনি, null ফলাফলটা আগেই বৈধ ফলাফল হিসেবে
   pre-register করা ছিল, আর কেন হলো না তার কারণটাও আমরা ব্যাখ্যা করেছি।

### ১.৩ টাইটেলের মানে

> **A Hybrid SHAP–LIME Consensus Method and Its Limits: A Reproducible Multi-Model
> Benchmark on Clinical Tabular Data**

| অংশ | কী বোঝাচ্ছে |
|---|---|
| Hybrid … Consensus Method | আমাদের fidelity-weighted rank fusion rule |
| and Its Limits | ১৮টা cell-এর একটাতেও এটা SHAP/LIME-এর ভালোটাকে হারাতে পারেনি |
| Reproducible | প্রতিটা সংখ্যা, figure, table কোড দিয়ে তৈরি; `make all` দিলে পুরো study আবার তৈরি হয় |
| Multi-Model | ছয় ধরনের classifier — আগের প্রায় সব তুলনামূলক কাজ এই axis-টাই fixed রাখে |
| Clinical Tabular Data | তিনটা public, de-identified মেডিকেল table (কোনো মেডিকেল দাবি আমরা করছি না) |

---

## ২. দরকারি শব্দগুলো, একবারেই

| শব্দ | সহজ মানে |
|---|---|
| **Tabular data** | সাধারণ টেবিল — row মানে রোগী, column মানে পরিমাপ। ছবি না, টেক্সট না। |
| **Classifier** | যে model শ্রেণি বলে দেয় — এখানে "রোগ আছে" বা "নেই", সাথে একটা probability। |
| **Black box** | যে model-এর যুক্তি তার parameter দেখে পড়া যায় না। Tree ensemble-রা এর প্রধান উদাহরণ — আর tabular data-তে এরাই সবচেয়ে ভালো কাজ করে। |
| **Post-hoc explanation** | model train হওয়ার **পরে**, আলাদা একটা টুল দিয়ে তৈরি ব্যাখ্যা — model-কে বদলানো হয় না। |
| **Feature attribution** | প্রতিটা feature-এর জন্য একটা সংখ্যা, যা বলে এই একটা prediction-কে সে কতটা ঠেলেছে। চিহ্ন: φ (ফাই)। |
| **Ranking / top-5 list** | φ-এর মান (চিহ্ন বাদ দিয়ে) অনুযায়ী feature সাজানো, বড় থেকে ছোট। practitioner আসলে top-5 তালিকাটাই পড়ে। |
| **Cell** | একটা (dataset, model) জোড়া। ৩ dataset × ৬ model = **১৮টা cell**। পেপারে শব্দটা বারবার আসে। |
| **Instance** | একটা row = একজন রোগী = একটা prediction, যেটা আমরা ব্যাখ্যা করেছি। মোট ১,৭৪৬টা। |

### ২.১ SHAP — এক লাইনের ধারণা

ভাবো, একটা টিম পুরস্কার জিতেছে আর তোমাকে টাকাটা খেলোয়াড়দের মধ্যে **ন্যায্যভাবে** ভাগ করতে হবে।
Game theory-তে এর একটাই ন্যায্য উত্তর আছে — **Shapley value**: প্রত্যেক খেলোয়াড়ের অবদান হিসাব
করা হয় সব সম্ভাব্য ক্রমে সে দলে যোগ দিলে কী যোগ হতো, তার গড় নিয়ে। SHAP **feature**-গুলোকে
খেলোয়াড় আর **prediction**-কে পুরস্কার ধরে। Tree model-এর জন্য `TreeExplainer` এটা tree-এর গঠন
কাজে লাগিয়ে **হুবহু এবং দ্রুত** হিসাব করে — তাই **tree-তে SHAP deterministic**: দুবার চালালে
বিট-টু-বিট একই সংখ্যা আসে।

### ২.২ LIME — এক লাইনের ধারণা

LIME পুরো model নিয়ে মাথা ঘামায় না। সে একটা row নেয়, তার হাজারখানেক সামান্য বদলানো কপি
বানায়, প্রতিটার জন্য model-এর কাছে prediction চায়, কপিগুলোকে মূল row-এর কাছাকাছি হওয়ার ভিত্তিতে
weight দেয়, আর তারপর ওই ছোট এলাকাটায় একটা **সরলরেখার model** ফিট করে। সেই সরলরেখার
coefficient-গুলোই ব্যাখ্যা। কপিগুলো random-ভাবে তৈরি হয়, তাই **LIME stochastic** — দুবার চালালে
সংখ্যা কিছুটা বদলায়।

### ২.৩ দুজন কেন একমত হয় না

কারণ দুজন আসলে **দুটো আলাদা প্রশ্নের** উত্তর দিচ্ছে। SHAP model-এর **output**-কে ভাগ করে দেয়;
LIME model-এর **স্থানীয় ফাংশনটাকে** approximate করে। তার উপর LIME random sample নেয়, আর তার
উত্তর নির্ভর করে তুমি কত sampling budget দিলে তার উপর। তাই দুটো আলাদা top-5 তালিকা কোনো bug না
— এটাই স্বাভাবিক। আসল প্রশ্ন হলো, এখন কী করব।

---

## ৩. এই কাজটা কেন দরকার ছিল

Tabular data-তে সবচেয়ে ভালো model-গুলো পড়া যায় না, অথচ এগুলোই এমন জায়গায় ব্যবহার হচ্ছে যেখানে
সিদ্ধান্ত মানুষের উপর প্রভাব ফেলে। আইনও এসে গেছে — EU AI Act উচ্চ-ঝুঁকির system-এর জন্য
interpretability বাধ্যতামূলক করেছে। মানে explanation এখন ঐচ্ছিক না। কিন্তু কেউ যদি SHAP আর LIME
দুটোই চালিয়ে দুটো আলাদা উত্তর পায়, literature তাকে বলে দেয় না কোনটা বিশ্বাস করবে।

আগের তুলনামূলক কাজগুলো এই প্রশ্নের মীমাংসা করতে পারে না, কারণ **তারা অনেক কিছু fixed রাখে**:

- সাধারণত একটাই model family, বা একটাই dataset — ফলে বোঝা যায় না ফলাফলটা explainer-এর গুণ
  নাকি ওই নির্দিষ্ট model-এর;
- "fidelity"-র সংজ্ঞা প্রতি পেপারে আলাদা — তাই ফলাফল তুলনাযোগ্য না;
- কোনো hypothesis test নেই, effect size নেই;
- **random-ranking control নেই** — ফলে fidelity স্কোরের কোনো scale নেই;
- এবং attribution কোন **output space**-এ (probability না log-odds) আছে, সেটা কেউ বলে না — যা
  চুপচাপ পুরো তুলনাটাই অবৈধ করে দেয় (পার্ট ৮ দেখো)।

> **যে বাক্যটা বলবে:** *ফাঁকটা কোনো অনুপস্থিত algorithm-এর না, ফাঁকটা একটা নিয়ন্ত্রিত তুলনার —
> আর সেটা একটা experiment দিয়েই পূরণ করা সম্ভব।*

---

## ৪. আমরা যা বানিয়েছি: hybrid method

টাইটেল এই অংশটা দিয়েই শুরু হয়, তাই এটা নোট ছাড়া ব্যাখ্যা করতে পারা দরকার।

### ৪.১ মূল ধারণা

SHAP আর LIME-এর সংখ্যাগুলো আলাদা scale-এ থাকে, তাই সরাসরি গড় করা যায় না। কিন্তু দুজনেই শেষে
**একই feature-গুলোর একটা সাজানো তালিকা** দেয়। **Rank** সরাসরি তুলনাযোগ্য — তাই আমরা rank-এর
স্তরে মেশাই। কোনো rescaling লাগে না, আর দুটো সংখ্যার scale একই মানে ধরে নিতেও হয় না।

### ৪.২ তিনটা rule — দুটো ধার করা, একটা আমাদের

| Rule | কীভাবে কাজ করে | কার |
|---|---|---|
| **Mean-rank (Borda)** | দুটো normalised rank-এর গড় | published baseline |
| **Reciprocal-rank fusion (RRF)** | প্রতিটা তালিকা দেয় 1/(60 + তার rank); যোগ করে আবার সাজানো | published baseline (Cormack et al., search engine-এ standard) |
| **Fidelity-weighted fusion** | দুটো rank-এর weighted গড়, যেখানে weight = প্রতিটা explainer-এর **মাপা** faithfulness | **আমাদের** |

RRF কেন দরকার: `1/(60+rank)` তালিকার নিচের দিকে গিয়ে চ্যাপ্টা হয়ে যায়, তাই একটা তালিকা খুব ভুল
হলেও সাধারণ গড়ের চেয়ে কম ক্ষতি করে।

### ৪.৩ "Fidelity-weighted" মানে কী — উদাহরণ দিয়ে

Mean-rank দুজনকে সমান বিশ্বাসযোগ্য ধরে। আমাদের rule ধরে না। এটা প্রত্যেক explainer-কে তার weight
**অর্জন** করতে বলে:

```
w = AOPC_shap / (AOPC_shap + AOPC_lime)        ← calibration split-এ মাপা
fused rank = w × (SHAP-এর rank) + (1 − w) × (LIME-এর rank)
```

AOPC হলো faithfulness-এর স্কোর (পার্ট ৬.৩) — একটা explainer যে feature-গুলোকে আগে রেখেছে,
সেগুলো মুছে দিলে model-এর prediction কতটা পড়ে যায়। মানে, যার ranking সত্যিই model-কে নাড়ায়,
fused rank-এ তার ভাগ বেশি।

ধরো এক cell-এ calibration-এ SHAP পেল 0.26 আর LIME পেল 0.22, তাহলে `w = 0.26 / 0.48 = 0.54`:

| Feature | SHAP rank | LIME rank | Mean-rank | Fidelity-weighted (w = 0.54) | চূড়ান্ত ক্রম |
|---|---|---|---|---|---|
| Glucose | 1 | 2 | 1.5 | 0.54·1 + 0.46·2 = **1.46** | ১ম |
| BMI | 3 | 1 | 2.0 | 0.54·3 + 0.46·1 = **2.08** | ২য় |
| Age | 2 | 4 | 3.0 | 0.54·2 + 0.46·4 = **2.92** | ৩য় |
| Insulin | 4 | 3 | 3.5 | 0.54·4 + 0.46·3 = **3.54** | ৪র্থ |

লক্ষ করো weighting কী করল: mean-rank শুধু গড়ের জোরে BMI-কে Age-এর আগে বসায়, আর
fidelity-weighting SHAP-এর মতের দিকে হেলে পড়ে, কারণ এই cell-এর calibration data-তে SHAP বেশি
weight অর্জন করেছে। Weight **প্রতিটা cell-এর জন্য আলাদা করে** হিসাব হয়, globally না — কারণ
faithfulness cell থেকে cell-এ বদলায়।

### ৪.৪ সততার রক্ষাকবচ: আলাদা calibration split

`w` fit করা হয় ব্যাখ্যা করা instance-এর **৩০%** দিয়ে, আর RQ6 পরীক্ষা করা হয় বাকি **৭০%**-এর
উপর। এই দুই সেট কখনো একে অন্যের সাথে মেশে না।

এটা যতটা সহজ শোনায়, ততটা তুচ্ছ না। যদি আমরা `w`-কে ঠিক সেই row-গুলো দিয়েই fit করতাম যেগুলোতে
পরে পরীক্ষা করেছি, তাহলে rule-টা **গঠনগতভাবেই** জিতে যেত — নিজের test set-এ tune করা হয়েছে,
সেই জয় কিছুই প্রমাণ করত না। প্রতিটা cell-এর split `results/consensus_splits.csv`-এ লেখা আছে,
তাই আলাদা রাখাটা শুধু দাবি না — যাচাইযোগ্য।

> স্যার hybrid নিয়ে একটাই technical প্রশ্ন করলে সম্ভবত এটাই করবেন। উত্তর:
> **weight যে data-তে fit করা, hypothesis সেই data-তে পরীক্ষা করা হয়নি।**

### ৪.৫ আমরা যে মানদণ্ড ঠিক করেছিলাম

Fused ranking-কে **একই cell-এ SHAP আর LIME-এর মধ্যে যেটা ভালো, সেটাকে** হারাতে হবে — দুজনের
গড়কে না, দুর্বলটাকেও না।

এত কঠিন কেন: গড়কে হারানো তুচ্ছ ব্যাপার। একটা শক্তিশালী আর একটা দুর্বল signal-এর যেকোনো
যুক্তিসঙ্গত মিশ্রণ ওই দুইয়ের মাঝামাঝির উপরে পড়ে, তাই "আমরা SHAP আর LIME-এর গড়কে হারিয়েছি"
কথাটার কোনো ব্যবহারিক মূল্য থাকত না। একজন practitioner hybrid ব্যবহার করবে কেবল তখনই, যখন সেটা
তার **হাতে থাকা সবচেয়ে ভালো** অপশনকে হারায়। এই মানদণ্ডটা কিছু চালানোর আগেই protocol-এ লেখা ছিল।

---

## ৫. এক্সপেরিমেন্টের নকশা

### ৫.১ তিনটা dataset, কঠিন-থেকে-সহজ ধারা মাথায় রেখে বাছা

| Dataset | Row | Feature | Positive rate | ROC-AUC | ভূমিকা |
|---|---|---|---|---|---|
| PIMA Indian Diabetes | 768 | 8 | 34.9 % | 0.83 | কঠিন task |
| Heart Disease (Cleveland) | 303 | 13 | 45.9 % | 0.90 | মাঝারি task |
| Chronic Kidney Disease | 400 | 24 | 62.5 % | 1.00 | সহজ task |

তিনটাই public, de-identified, পুনর্বিতরণযোগ্য (CC0 বা CC BY 4.0), এবং repository-তে SHA-256
checksum সহ রাখা — প্রতিবার চালানোর সময় checksum যাচাই হয়।

**তিনটা কেন, আর ভিন্ন কঠিনতার কেন:** যাতে জিজ্ঞেস করা যায় — explainer-দের মধ্যে মিল কি decision
boundary কত কঠিন তার উপর নির্ভর করে? একটা dataset দিয়ে এই প্রশ্নটাই করা যায় না। (আমরা যা
পেয়েছি সেটা প্রত্যাশার চেয়ে বেশি মজার — পার্ট ৭.২।)

**রোগগুলো এখানে শুধু benchmark।** আমরা কোনো মেডিকেল দাবি করছি না, এখানকার কোনো model
রোগনির্ণয়ের জন্য ব্যবহারযোগ্য না, আর কোনো feature ranking-কে clinical বা causal বক্তব্য হিসেবে
পড়া যাবে না। কথাটা শুরুতেই বলে দেবে — এটা পুরো presentation-কে রক্ষা করে।

**আগেই ঠিক করা data cleaning:** PIMA-তে glucose, blood pressure, skin thickness, insulin আর
BMI-তে শূন্য জৈবিকভাবে অসম্ভব, তাই সেগুলো missing ধরা হয়েছে — কিন্তু *pregnancies*-এ শূন্য বৈধ,
তাই রাখা হয়েছে। Heart-এ পাঁচ-স্তরের target-কে > 0 ধরে দুই ভাগ করা হয়েছে, আর nominal code-গুলো
category হিসেবে রাখা হয়েছে যাতে linear model তাদের মধ্যে কোনো ক্রম ধরে না নেয়।

### ৫.২ ছয়টা model

Logistic regression, decision tree, random forest, XGBoost, LightGBM, আর একটা stacking
ensemble — linear → একক tree → bagged → boosted → stacked, পুরো ধারাটা ঢেকে।

**Stacking কেন আছে:** এটাই একমাত্র family যার জন্য tree-নির্দিষ্ট SHAP backend নেই, তাই SHAP-কে
sampling-ভিত্তিক explainer-এ ফিরতে হয়। ফলে stacking-ই একমাত্র জায়গা যেখানে **SHAP-এর নিজের
run-to-run variance মাপার মতো একটা ব্যাপার হয়**। এটা নকশায় আছে একটা নির্দিষ্ট কারণে, শুধু
সংখ্যা বাড়াতে না।

Logistic regression আর decision tree আছে interpretable রেফারেন্স পয়েন্ট হিসেবে, প্রতিযোগী
baseline হিসেবে না।

### ৫.৩ ১৮টা cell, ১,৭৪৬টা ব্যাখ্যা করা prediction

প্রতি cell-এ সর্বোচ্চ ১৫০টা test instance ব্যাখ্যা করা হয়: PIMA ১৫০ (১৫৪-র মধ্যে), Heart ৬১
(সবগুলো), CKD ৮০ (সবগুলো)। তাই 6 × (150 + 61 + 80) = **১,৭৪৬**।

### ৫.৪ দুটো আলাদা track — leakage আটকানোর ব্যবস্থা

| | কী করে |
|---|---|
| **Track A** | nested cross-validation দিয়ে performance মাপে: ৫টা outer fold হিসাবের জন্য, ৩টা inner fold hyperparameter বাছার জন্য; imputation, scaling, encoding সব pipeline-এর **ভিতরে**, তাই সেগুলো কেবল inner-training row দিয়ে fit হয়। |
| **Track B** | Track A-র জেতা hyperparameter নিয়ে একটা stratified 80/20 split-এ **একবার** refit করে, আর প্রতি cell-এ ওই একটা model-এর উপরেই সব explainer চালায়। |

**আলাদা রাখার কারণ:** explanation analysis-এর জন্য প্রতি cell-এ **একটা** model দরকার, পাঁচটা
fold-model না — "পাঁচটা model-এর গড়"-কে ব্যাখ্যা করা যায় না। আর explainer যে partition দেখছে,
সেটা tuning-এ অংশ নেওয়া চলবে না। এই দুটো গুলিয়ে ফেলাই explanation study-তে leakage-এর সবচেয়ে
প্রচলিত রূপ, এবং output দেখে সেটা ধরা যায় না।

### ৫.৫ SMOTE ব্যবহার করিনি, ইচ্ছে করেই

Class imbalance এখানে মৃদু (34.9–62.5 % positive)। Synthetic oversampling সেই data manifold
বদলে দিত যেখান থেকে LIME-এর sampler আর SHAP-এর background — দুজনেই — নমুনা নেয়। মানে, ঠিক যে
জিনিসটার তুলনা করছি, সেটাই বদলে যেত। আমরা যেখানে সম্ভব balanced class weight ব্যবহার করেছি আর
accuracy-র পাশে MCC ও PR-AUC রিপোর্ট করেছি। সিদ্ধান্তটা protocol-এ লেখা, যাতে এটা একটা নকশাগত
সিদ্ধান্ত হিসেবেই পড়া হয়, ভুলে বাদ পড়া হিসেবে না।

### ৫.৬ Random control

একটা সম্পূর্ণ random feature ranking-কেও আমরা একটা বাড়তি "explainer" হিসেবে চালিয়েছি। এটা
ফাঁকা জায়গা ভরাট না: এটা ছাড়া fidelity স্কোরের কোনো scale-ই থাকে না (পার্ট ৮, ফাঁদ ৪)।

---

## ৬. ছয়টা প্রশ্ন, আর প্রতিটার মাপকাঠি

| RQ | সহজ ভাষায় প্রশ্ন | প্রধান metric |
|---|---|---|
| **RQ1** | SHAP আর LIME কি feature একইভাবে সাজায়? আর তা model বা task-এর উপর নির্ভর করে কি? | প্রতি instance-এ Spearman ρ |
| **RQ2** | Explanation কতটা stable — তিন ধরনের stability আলাদা করে? | CV; local Lipschitz; cross-refit agreement |
| **RQ3** | Random ranking-এর তুলনায় explanation-গুলো কি সত্যিই model-এর কাজের সাথে মেলে? | AOPC (deletion test) |
| **RQ4** | প্রতিটা explanation-এর সময়ের খরচ কত, আর তার বিনিময়ে কত faithfulness পাওয়া যায়? | প্রতি explanation-এ median সেকেন্ড |
| **RQ5** | উপরের feature-গুলো কি মেডিকেল literature যা বলে তার সাথে মেলে? | top-5 overlap (Jaccard@5) |
| **RQ6** | দুটো মেশালে কি একা ভালোটার চেয়ে ভালো হয়? | max(SHAP, LIME)-এর সাপেক্ষে ΔAOPC |

### ৬.১ Agreement (RQ1)

চারটা মাপ, কারণ প্রত্যেকটার একটা অন্ধ দিক আছে: **Spearman ρ** (পুরো ক্রম দুটো মেলে কি?),
**Kendall τ_b** (একই ধারণা, tie-সহনশীল), **Jaccard@k** (top-k **সেট** দুটো কতটা মেলে, k = 3 ও 5),
আর **rank-biased overlap (RBO)** (উপরের দিকে বেশি ওজন দেয়, আর কোনো নির্দিষ্ট k বাছতে হয় না)।

আমরা একটা সূক্ষ্ম সমস্যা ধরেছি ও ঠিক করেছি: যেহেতু দুটো ranking-ই **একই** feature সেটের
বিন্যাস, RBO-র একটা কঠিন floor আছে যেটা feature সংখ্যার সাথে বদলায়। একেবারে উল্টো ranking-ও
৮টা feature হলে 0.589 পায়, কিন্তু ২৪টা হলে মাত্র 0.193। তাই কাঁচা RBO ভিন্ন প্রস্থের dataset-এর
মধ্যে তুলনাযোগ্য না — আমরা floor-normalised একটা রূপও রিপোর্ট করি। এটা বলার মতো: এতে বোঝা যায়
metric-গুলো শুধু ডাকা হয়নি, বোঝাও হয়েছে।

### ৬.২ Stability (RQ2), তিন ভাগে

এই ভাগ করাটাই contribution ২। তিনটা সত্যিই আলাদা প্রশ্ন:

১. **Re-run variance (2a) — "টুলটা কি reproducible?"** একই row ১০ বার ব্যাখ্যা করো ভিন্ন random
   seed দিয়ে, দেখো সংখ্যা কতটা নড়ে (coefficient of variation)।
২. **Local robustness (2b) — "রোগীর তথ্যে সামান্য পরিবর্তনে গল্পটা বদলায় কি?"** row-এর চারপাশে
   0.1 standard deviation দূরত্বে ৫টা প্রতিবেশী নাও (categorical column স্থির রেখে), আর input-এর
   প্রতি একক পরিবর্তনে explanation-এর সর্বোচ্চ পরিবর্তন মাপো (local Lipschitz estimate)। কম
   হওয়া ভালো।
৩. **Retraining stability (2c) — "Model আবার train করলে গল্পটা টেকে কি?"** একই training data-র
   ১০টা bootstrap resample-এ model আবার fit করো, দেখো global feature ranking টেকে কিনা।

**কেন এটা জরুরি:** যে study একটামাত্র "stability" সংখ্যা দেয়, সে চুপচাপ এই তিনটার একটা বেছে
নিচ্ছে — আর সেই বাছাই conclusion পুরো বদলে দেয় (পার্ট ৭.৩)। Retraining stability শুধু তিনটা
ensemble model-এ চালানো হয়েছে, যেটা compute-এর কারণে **আগেই** ঠিক করা ছিল।

### ৬.৩ Fidelity (RQ3) — deletion test আর AOPC

ধারণাটা: explanation যদি faithful হয়, তাহলে সে যে feature-গুলোকে **আগে** রেখেছে, সেগুলোই
model-এর সবচেয়ে বেশি দরকারি হওয়া উচিত। তাই:

১. explanation-এর ranking নাও;
২. top-1 feature মুছে দাও (training median বসিয়ে) আর আবার predict করো;
৩. model-এর **নিজের predicted class**-এর probability কতটা পড়ল, লিখে রাখো;
৪. top-2, top-3, … সর্বোচ্চ ১০টা পর্যন্ত একইভাবে করো (dataset-এ feature কম হলে সবগুলো);
৫. এই পতনগুলোর গড় নাও। এই গড়টাই **AOPC**।

AOPC বেশি = বেশি faithful। এর পাশাপাশি insertion AUC, comprehensiveness আর sufficiency-ও রিপোর্ট
করা হয়েছে, এবং পুরো জিনিসটা আরেকটা ভিন্ন প্রতিস্থাপন মান দিয়ে (median-এর বদলে marginal draw)
আবার চালানো হয়েছে sensitivity check হিসেবে — কারণ median বসালে এমন row তৈরি হয় যা বাস্তবে
থাকতে পারে না, এটা একটা পরিচিত সমস্যা।

### ৬.৪ Cost (RQ4)

Explainer কল-এর চারপাশে single-threaded wall-clock সেকেন্ডের median। সরল, আর এটা মেশিন-নির্ভর —
সেটা আমরা স্বীকার করেছি।

### ৬.৫ Plausibility (RQ5)

প্রত্যেক explainer-এর top-5 আর প্রকাশিত clinical literature অনুযায়ী গুরুত্বপূর্ণ feature-এর একটা
তালিকার মধ্যে overlap — তালিকাটা citation সহ তৈরি এবং **কোনো explanation হিসাব হওয়ার আগেই
freeze করা**। এটা কেবল সমর্থক প্রমাণ: literature-ভিত্তিক তালিকা অসম্পূর্ণ, ground truth না।
আমরা এটা আগেই বলে রেখেছিলাম।

### ৬.৬ Statistics, এক প্যারাগ্রাফে

এগারোটা hypothesis আগেই লিখে রাখা হয়েছিল — প্রত্যেকটার test, দিক, unit of analysis আর family সহ।
সব test non-parametric — সর্বত্র Wilcoxon signed-rank, শুধু একটায় Mann–Whitney U আর আরেকটায়
১০,০০০ draw-এর permutation test। *p*-value প্রত্যেক family-র **ভিতরে Holm correction** করা।
প্রতিটা test-এর সাথে **effect size** আর প্রতিটা estimate-এর সাথে **৯৫% bootstrap confidence
interval** আছে — কারণ প্রতি cell-এ শত শত instance থাকলে "statistically significant" হওয়াটা সস্তা,
আসল ফলাফল হলো effect size। একটা জিনিস জানা দরকার: RQ1 পরীক্ষা করা হয়েছে ρ > 0.50-এর বিরুদ্ধে,
শূন্যের বিরুদ্ধে না — শূন্যের বিরুদ্ধে পরীক্ষা করা straw man হতো — আর 0.50 সীমাটা কোনো ρ হিসাব
হওয়ার আগেই ঠিক করা ছিল।

---

## ৭. আমরা যা পেয়েছি

### ৭.১ RQ1 — যথেষ্ট মিল আছে, কিন্তু একটার বদলে আরেকটা চালানো যায় না

- Median Spearman ρ = **0.74**
- Median top-5 Jaccard = **0.67**
- ρ > 0.50 মানদণ্ড পার করেছে **১৮টার মধ্যে ১৫টা cell**-এ

**0.67 মানে ঠিক কী, কারণ এটা জিজ্ঞেস করা হতে পারে:** median-এ দুটো top-5 তালিকার **৫টার মধ্যে
৪টা feature একই, ১টা আলাদা**। (পেপারে অমিলটাকে বলা হয়েছে "প্রায় এক-তৃতীয়াংশ", যা Jaccard
*distance*, অর্থাৎ 1 − 0.67। কেউ যদি feature-এর হিসাবে জানতে চায়, উত্তর: median-এ পাঁচের মধ্যে
একটা আলাদা; union-এর ছয়টার মধ্যে চারটা মেলে।)

**কেন গুরুত্বপূর্ণ:** disagreement-এর literature পড়ে যতটা অমিল আশা করা যায়, তার চেয়ে দুজনের
মিল বেশি — আর তবুও অমিলটা এত যে দুটো একসাথে চালানো কোনো **validation check না**। শুধু যে
feature-গুলোতে দুজন একমত সেগুলো নিলে, প্রত্যেক তালিকা থেকে একটা feature চুপচাপ বাদ পড়ে যায়।
Top-5-এ কে আছে তার উপর সিদ্ধান্ত নির্ভর করলে, কোন explainer বাছা হলো সেটাও সিদ্ধান্তের অংশ, আর
তা নথিতে থাকা উচিত।

### ৭.২ চমক: মিল সবচেয়ে কম **সবচেয়ে সহজ** dataset-এ

| Dataset | ROC-AUC (কতটা সহজ) | Median ρ (কতটা মিল) |
|---|---|---|
| PIMA | 0.83 (কঠিন) | 0.76 |
| Heart | 0.90 (মাঝারি) | 0.78 |
| CKD | 1.00 (সবচেয়ে সহজ) | **0.63** |

বেশিরভাগ মানুষের ধারণা — task কঠিন হলে অমিল বাড়বে — সেটা ঘটেনি।

**আমাদের ব্যাখ্যা:** মিল নিয়ন্ত্রণ করে কঠিনতা না, **feature redundancy**। CKD-তে একাধিক feature
একা একাই প্রায় যথেষ্ট class আলাদা করার জন্য। এমন হলে দুটো explainer **দুজনেই** স্থানীয়ভাবে
faithful হতে পারে, অথচ আলাদা feature-এর নাম বলতে পারে — কারণ একই prediction ব্যাখ্যা করার
একাধিক সমান ভালো পথ আছে।

**এখানে সাবধান, আর এটা নিজেই বলে দেবে:** H1c (মিল কি task difficulty অনুসরণ করে?) **reject
হয়নি** — adjusted *p* = 0.243। তিনটা dataset দিয়ে dataset-জুড়ে trend পরীক্ষা করার statistical
power-ই নেই। তাই এটা একটা **বর্ণনামূলক পর্যবেক্ষণ, সাথে একটা প্রস্তাবিত ব্যাখ্যা** — প্রমাণিত
নিয়ম না। সীমাটা নিজে বলে দেওয়া ধরা পড়ার চেয়ে অনেক ভালো।

### ৭.৩ RQ2 — "reproducible" আর "robust" এক জিনিস না

এই ফলাফলটাই আমরা সবচেয়ে বেশি এগিয়ে নিতে চাইব, আর তিনটা সংখ্যা একসাথেই অর্থপূর্ণ:

| Stability-র ধরন | SHAP | LIME | পাঠ |
|---|---|---|---|
| Re-run (একই model, নতুন seed) | **ঠিক 0** variance | CV 0.68 | tree-তে SHAP বিট-টু-বিট একই; LIME না |
| Input perturbation (local Lipschitz, কম ভালো) | **0.18** | 0.65 | SHAP স্পষ্টতই বেশি robust, ১৮টার মধ্যে ১৫টা cell-এ পার্থক্য — কিন্তু কেউই অসংবেদনশীল না |
| Model retraining (cross-refit rank agreement) | **0.83** | 0.76 | একই data-র resample-এ model refit করলে **দুজনেরই** ranking নড়ে |

দুটো জিনিস নিয়ে নিখুঁত থাকতে হবে:

- SHAP-এর শূন্য re-run variance **আগেই protocol-এ method-এর একটা ধর্ম হিসেবে ঘোষণা করা ছিল**,
  আবিষ্কার হিসেবে দেখানো হয়নি। নির্দিষ্ট background দিলে TreeExplainer deterministic। এটাকে
  আবিষ্কার বলে দাবি করা অসৎ হতো, আর আমরা করিনি।
- LIME-এর reproducibility **budget-এর সাথে বাড়ে** — `num_samples` বাড়ালে variance কমে। মানে
  LIME-এর reproducibility ব্যবহারকারীর কেনা একটা জিনিস, বিনামূল্যে পাওয়া ধর্ম না।

**এই ভাগ করার আসল কথা:** যে study শুধু প্রথম উপায়ে stability মাপে, সে SHAP-কে **সম্পূর্ণ stable**
লিখে থেমে যেত। কিন্তু তার cross-refit agreement 0.83 — একই training data-র bootstrap resample-এ
model আবার fit করলে global ranking নড়ে যায়। **অস্থিরতাটা model-এর ভিতরে, explainer-এর ভিতরে
না — আর attribution method যত deterministic হোক, তা এটা সরাতে পারে না।** কেউ explanation
stability রিপোর্ট করলে তার বলা উচিত সে এই তিনটার কোনটা মেপেছে।

### ৭.৪ RQ3 ও RQ4 — SHAP একইসাথে বেশি faithful আর বেশি সস্তা

| | SHAP | LIME | Random control |
|---|---|---|---|
| AOPC (faithfulness, বেশি ভালো) | **0.260** | 0.224 | 0.075 |
| প্রতি explanation-এ সেকেন্ড | **0.0019** | 0.0462 | — |

- Holm correction-এর পরেও দুজনেই random control-কে হারিয়েছে **৩৬টার মধ্যে ৩৬টা** তুলনায়। এটাই
  0.260 আর 0.224-কে অর্থপূর্ণ করে।
- গতির ব্যবধান প্রায় **24.8×**, আর এটা **গঠনগত, implementation-এর খুঁটিনাটি না**: TreeSHAP
  tree-এর গঠন কাজে লাগায়, আর LIME প্রতিটা explanation-এর জন্য হাজার হাজার বদলানো row-তে
  model চালাতে বাধ্য।
- Marginal-baseline sensitivity check ক্রমটা উল্টায় না — এটা জরুরি, কারণ median বসানো
  instance-গুলোকে data manifold-এর বাইরে ঠেলে দেয়।

**যে conclusion বলবে:** যেখানেই tree backend আছে, SHAP **একইসাথে** বেশি faithful আর বেশি সস্তা।
LIME-এর পক্ষে বাকি থাকে **model-agnosticism** — সে যেকোনো model-এ কাজ করে — আর এটা সত্যিকারের
সুবিধা, ছোট কিছু না। আমাদের stacking cell-গুলো, যেখানে tree explainer নেই, সেখানেই দেখা যায়
এর দাম কত।

### ৭.৫ RQ5 — plausibility check পার্থক্য ধরতে পারেনি

Literature reference set-এর সাথে overlap-এ দুজনেই random control-কে স্পষ্টভাবে হারায়, কিন্তু
দুজনের মধ্যে পার্থক্য দেখা যায় না।

আমরা এটাকে **metric সম্পর্কে null ফলাফল** হিসেবে রিপোর্ট করেছি, method সম্পর্কে না:
literature থেকে তৈরি reference set অসম্পূর্ণ এবং ground truth না, আর ৮ feature-এর dataset-এ
top-5 overlap-এর নড়াচড়ার জায়গা খুব কম। RQ5-কে study-র সবচেয়ে দুর্বল মাপ হিসেবে আগেই
pre-register করা হয়েছিল, এবং চালানোর আগেই আমরা সেটা বলেছি।

### ৭.৬ RQ6 — hybrid মানদণ্ড পার করতে পারেনি

- Fusion SHAP/LIME-এর ভালোটাকে হারিয়েছে **১৮টার মধ্যে ০টা cell**-এ
- Median rank-biserial effect **−0.80** — বড়, এবং ধারাবাহিকভাবে ভুল দিকে
- সবচেয়ে ভালো fusion rule-এর AOPC **0.248**, আর একা SHAP-এর **0.260**

**কেন ব্যর্থ হলো — এটা আত্মবিশ্বাসের সাথে বলবে, এটাই উত্তরের সবচেয়ে শক্ত অংশ:**

Fusion কাজ করে যখন দুটো শর্ত একসাথে থাকে: input-গুলো আলাদাভাবে noisy, **এবং** তাদের ভুলগুলো
পরস্পর সম্পর্কহীন। এটাই ensemble-এর পুরো যুক্তি। এখানে **একটাও** নেই।

১. দুই explainer-এর **যথেষ্ট মিল** আছে (ρ = 0.74), তাই fusion-এর উদ্ধার করার মতো স্বাধীন signal
   প্রায় নেই।
২. একজন **ধারাবাহিকভাবে ভালো**, তাই তার rank-কে একটা দুর্বল ও অনেকটা অপ্রয়োজনীয় signal-এর
   সাথে গড় করলে সেটা কেবল পাতলা হয়।

আমাদের fidelity-weighted rule ঠিক ২ নম্বর সমস্যাটার বিরুদ্ধেই বানানো হয়েছিল — বেশি faithful
explainer-কে বেশি weight দিয়ে — এবং **তিনটা rule-এর মধ্যে সাধারণত এটাই সবচেয়ে ভালো করেছে**।
কিন্তু প্রতি cell-এ ছোট একটা calibration sample থেকে শেখা weight ব্যবধানটা মেটাতে পারে না।

**সাধারণ শিক্ষা, যা এই থিসিসের বাইরেও প্রযোজ্য:** explanation aggregation-এর জন্য দরকার
**পরিপূরক (complementary)** base explainer। Tree model-এ SHAP আর LIME যথেষ্ট পরিপূরক না। এটা
বলে দেয় fusion আদৌ চেষ্টা করার মতো কখন — আর এটাই negative ফলাফল থেকে পাওয়া কাজের জিনিস।

### ৭.৭ এগারোটা hypothesis, সহজ ভাষায়

| H | কী বলে | ফলাফল |
|---|---|---|
| H1 | SHAP/LIME-এর মিল ρ = 0.50-এর চেয়ে বেশি | **১৫/১৮** cell-এ null reject |
| H1b | simple model আর ensemble-এ মিল আলাদা | reject হয়নি (p = 0.291) |
| H1c | মিল task difficulty অনুসরণ করে | reject হয়নি (p = 0.243) |
| H2a | LIME-এর re-run variance শূন্যের বেশি | **১৮/১৮** |
| H2b | local robustness-এ দুজন আলাদা | **১৫/১৮** (SHAP বেশি robust) |
| H2c | retraining stability-তে দুজন আলাদা | **৯/৯** |
| H3a | fidelity-তে দুজন আলাদা | **১৩/১৮** (SHAP বেশি faithful) |
| H3b | দুজনেই random control-কে হারায় | **৩৬/৩৬** |
| H4 | SHAP-এর খরচ LIME-এর চেয়ে কম | **১৫/১৮** |
| H5 | plausibility overlap-এ দুজন আলাদা | reject হয়নি (p = 0.375) — exploratory |
| H6 | fusion দুজনের ভালোটাকে হারায় | **০/১৮** — pre-registered null ফলাফল |

"Reject" সবসময় মানে ওই কতগুলো cell-এ Holm-adjusted p < 0.05-এ **null** hypothesis reject হয়েছে।
চারটা hypothesis reject হয়নি। এর মধ্যে তিনটা — H1b, H1c, H5 — নকশাগতভাবেই underpowered: এরা
প্রতি cell-এ একটা মান নিয়ে অল্প কয়েকটা cell-এ তুলনা করে, তাই ওখানে null মানে "আমরা বলতে
পারলাম না", "কোনো প্রভাব নেই" না। H6-ই মূল একটা, আর সেখানে প্রমাণ জোরালো ও এক দিকের: fusion
প্রতিটা cell-এ হেরেছে।

---

## ৮. চারটা ফাঁদ — এখানেই আসল নম্বর

এই চারটার প্রত্যেকটা SHAP-বনাম-LIME তুলনাকে চুপচাপ অবৈধ করে দিতে পারে, আর **ভুল হলে output
দেখে একটাও ধরা যায় না**। স্যার যদি জিজ্ঞেস করেন এটা একটা class project থেকে আলাদা কীভাবে, উত্তর
এটাই।

### ফাঁদ ১ — দুজনকে **একই রাশি** ব্যাখ্যা করতে হবে

LIME ব্যাখ্যা করে predicted **probability**। SHAP-এর TreeExplainer boosted model-এ default-ভাবে
**log-odds margin** ফেরত দেয়। সরাসরি তুলনা করলে তুমি দুটো আলাদা রাশিকে দুটো আলাদা scale-এ
মিলাচ্ছ — আর সংখ্যাগুলো দেখতে একেবারে স্বাভাবিকই লাগবে।

**আমরা যা করেছি:** সর্বত্র probability-space attribution বাধ্যতামূলক করেছি, আর flag-এর উপর ভরসা
না করে সেটা **যাচাই** করেছি — পরীক্ষা করে দেখেছি `base value + সব attribution-এর যোগফল` model-এর
`predict_proba` মিলিয়ে দেয় কিনা। Cell-জুড়ে median additivity error **3.5 × 10⁻⁸**। ১৮টার মধ্যে
একটা cell pre-registered gate পার করেনি; পার্ট ১০ দেখো।

### ফাঁদ ২ — reference distribution-কে আসল data হতে হবে

SHAP-এর background set-কে k-means দিয়ে সংক্ষেপ করা খুব প্রচলিত। কিন্তু তাতে **কৃত্রিম বিন্দু**
তৈরি হয়, কারণ one-hot column-এর গড় নিলে ভগ্নাংশ আসে যা model কখনো দেখেনি। এমন off-manifold
reference-এর বিরুদ্ধে interventional TreeSHAP চালালে সে নিজের additivity গ্যারান্টিই ভাঙে।

**আমরা ক্ষতিটা মেপেছি:** Heart/LightGBM cell-এ k-means দিয়ে additivity error **9.9 × 10⁻³**, আর
আসল training row দিয়ে **7 × 10⁻⁹**। তাই background হলো **আসল training row**-এর একটা seeded,
class-stratified উপ-নমুনা — যা tree SHAP-কে deterministic-ও রাখে।

### ফাঁদ ৩ — one-hot encoding একটা feature-কে কয়েকটা model input-এ ভেঙে ফেলে

একটা মূল feature কয়েকটা model column হয়ে যেতে পারে। যোগ না করলে ভিন্ন model-এর ranking ভিন্ন
শব্দভাণ্ডারের উপর হতো, আর "top-5 feature মুছে দাও" মানে হতো পাঁচটা **column** মোছা — ফলে ২৫
column-এ ছড়ানো ১৩-feature dataset কম perturb হতো, আর ৮-feature dataset প্রায় অটুট থাকত।
Dataset-এর মধ্যে তুলনাটাই চুপচাপ অন্যায্য হয়ে যেত।

**আমরা যা করেছি:** ranking করার আগেই প্রত্যেক feature-এর column-গুলোর attribution যোগ করা হয়,
আর perturbation পুরো feature group-এর উপর চলে। Column-থেকে-feature map processed data-র সাথে
লেখা হয় এবং unit-test করা।

### ফাঁদ ৪ — control ছাড়া fidelity স্কোরের কোনো scale নেই

আটটা feature-এর **যেকোনো** কয়েকটা মুছলেই prediction নড়ে। তাই একা 0.31 AOPC-র কোনো ব্যাখ্যা নেই
— বোঝার উপায় নেই এটা ভালো explanation-এর ফল, নাকি শুধু তথ্য হারানোর ফল।

**আমরা যা করেছি:** প্রতিটা fidelity সংখ্যা একটা সম্পূর্ণ random ranking-এর সাপেক্ষে রিপোর্ট করা,
যেটা পায় **0.075**। এটাই SHAP-এর 0.260 আর LIME-এর 0.224-কে অর্থপূর্ণ করে।

---

## ৯. Pre-registration আর reproducibility

### ৯.১ Pre-registration মানে এখানে কী

প্রতিটা research question, hypothesis, metric-এর সূত্র, statistical test, significance level আর
correction পদ্ধতি `docs/PROTOCOL.md`-এ লিখে **নিজের একটা আলাদা commit-এ git-এ জমা দেওয়া হয়েছে,
কোনো ফলাফল-সহ commit হওয়ার আগে**। তাই repository-র history নিজেই প্রমাণ যে ফলাফল দেখে analysis
বদলানো হয়নি। Freeze-এর পরের যেকোনো পরিবর্তন তারিখ আর কারণ সহ একটা amendment log-এ আছে — কখনো
চুপচাপ edit করে না।

**এই থিসিসের জন্য কেন বিশেষভাবে জরুরি:** RQ6-এর negative ফলাফল বিশ্বাসযোগ্য কেবল তখনই, যদি আমরা
goalpost সরাতে না পারতাম। পারতাম না — কঠিন মানদণ্ড আর "contribution ৩-এ null ফলাফলও একটা বৈধ ও
রিপোর্টযোগ্য ফলাফল" — দুটোই কিছু চালানোর আগেই protocol-এ লেখা ছিল।

### ৯.২ এটা তিনবার নিজের দাম উঠিয়ে এনেছে

protocol দাবি করেছিল বলেই যে test-গুলো লেখা হয়েছিল, সেগুলো তিনটা সত্যিকারের ভুল ধরেছে — আর
**তিনটার একটাও ফলাফল দেখে ধরা যেত না**:

১. **Additivity check** একটা background-তৈরির ভুল ধরেছে, যেটা LightGBM cell-এ SHAP-এর নিজের
   additivity প্রায় এক শতাংশ ভাঙছিল, অথচ বাকি সব cell দেখতে সুস্থ ছিল।
২. **প্রত্যেক confidence interval-কে নিজের point estimate ধারণ করতে হবে** — এই test ধরেছে যে
   একটা BCa bootstrap একটা non-smooth statistic-এ ভুলভাবে প্রয়োগ হচ্ছিল।
৩. **RBO floor**-এর একটা test ধরেছে যে কাঁচা RBO ভিন্ন প্রস্থের dataset-এর মধ্যে তুলনাযোগ্য না।

তিনটাই protocol-এর amendment log-এ লেখা আছে।

### ৯.৩ এই repository-তে "reproducible" মানে ঠিক কী

- dependency exact version-এ pinned (`requirements.lock.txt`)
- source dataset SHA-256 checksum সহ repository-তে রাখা, আর **প্রতিবার চালানোর সময় যাচাই হয়** —
  না মিললে সেটা hard failure, warning না
- একটা manifest যেখানে প্রতিটা artefact-এর digest, git commit, seed set, resolved library
  version আর hardware লেখা থাকে
- **প্রতিটা সংখ্যা, figure আর table কোড দিয়ে তৈরি; একটাও হাতে লেখা না**
- global seed 42; repeat *r*-এর seed 1000 + *r*; bootstrap resample *b*-এর seed 2000 + *b*
- `make all` দিলে raw data থেকে typeset PDF পর্যন্ত পুরো study আবার তৈরি হয়

---

## ১০. সীমাবদ্ধতা — কেউ জিজ্ঞেস করার আগেই বলে দাও

- **তিনটা dataset, সবই ছোট, সবই clinical।** প্রতি cell-এর test-গুলোর যথেষ্ট statistical power
  আছে, কিন্তু **dataset-জুড়ে** সাধারণীকরণ মাত্র তিনটা বিন্দুর উপর দাঁড়ানো। কঠিনতার ধারাটা
  একটা নকশাগত ক্ষতিপূরণ, প্রশস্ততার বিকল্প না।
- **Perturbation baseline off-manifold।** Training median বসালে এমন row তৈরি হয় যা বাস্তব
  প্রক্রিয়া কখনো তৈরি করত না। একটা marginal-draw রূপ ক্রমটা উল্টায়নি, কিন্তু দুটোতেই একটা
  independence অনুমান আছে।
- **Plausibility reference set ground truth না।** Citation সহ literature থেকে তৈরি এবং কোনো
  ফলাফলের আগেই freeze করা হলেও, এগুলো অসম্পূর্ণ, non-causal, আর সূত্রগুলো যা মাপতে চেয়েছে তার
  দ্বারা প্রভাবিত। এজন্যই RQ5 কেবল সমর্থক প্রমাণ।
- **Retraining stability ছয়টার মধ্যে তিনটা model-এ** — compute-এর কারণে **আগেই** সীমিত করা,
  ফলাফল দেখে না।
- **Timing এক মেশিনের।** 24.8× ব্যবধানটা গঠনগত পার্থক্যের ফল, কিন্তু তার সঠিক মাপ
  hardware-নির্ভর।
- **কোনো human evaluation নেই।** এখানকার প্রতিটা metric model–explanation জোড়ার একটা ধর্ম।
  একজন চিকিৎসক explanation-টা কাজে লাগে মনে করেন কিনা, আমরা মাপিনি।
- **Black box-এর একটা faithful post-hoc explanation তবুও interpretable model না** (Rudin-এর
  যুক্তি, যেটা আমরা এড়িয়ে না গিয়ে উদ্ধৃত করেছি)।

### ১০.১ Additivity gate-এর ব্যর্থতা — এটা নিজেই বলে দাও

PIMA/LightGBM cell-এ ব্যাখ্যা করা **১৫০টার মধ্যে ৫টা** instance-এ `predict_proba` পুনর্গঠন
5 × 10⁻³ সহনসীমার চেয়ে খারাপ (সবচেয়ে খারাপ 0.094), অথচ ওই cell-এর median 1.1 × 10⁻⁹। মানে ওই
cell-এ pre-registered gate **পূরণ হয়নি**। অসঙ্গতিটা background-এর আকারের উপর নির্ভর করে না, আর
আমরা এর কারণ নির্ধারণ করতে পারিনি; সবচেয়ে সম্ভাব্য সন্দেহ — median দিয়ে ভরাট করা মানের সাথে
মিলে যাওয়া — পরীক্ষা করে বাতিল হয়েছে।

Protocol-এ আগেই লেখা দুটো বিকল্প ছিল: ওই cell-কে log-odds space-এ পাঠানো, যা তাকে প্রতিটা
cross-explainer তুলনা থেকে বাদ দিয়ে দিত; অথবা gate-টা ঢিলা করা। **আমরা একটাও করিনি।** যে gate
ফেল করেছে দেখে তারপর gate ঢিলা করা — pre-registration ঠিক এই জিনিসটা আটকাতেই আছে। Cell-টা রাখা
হয়েছে, ব্যর্থতাটা পেপারে লেখা হয়েছে, আর প্রতিটা hypothesis-এর হিসাব ওই cell বাদ দিয়েও আবার
রিপোর্ট করা হয়েছে — hypothesis টেবিলের কোনো সিদ্ধান্ত বদলায় না।

এটা নিজে বলা দুর্বলতা না, শক্তি: এতে দেখা যায় protocol সত্যিই কাজ করছে।

---

## ১১. পেপারের সেকশন-ভিত্তিক মানচিত্র

| Section | কী করে | এক বাক্যে যা বলবে |
|---|---|---|
| **Abstract** | পুরো study এক প্যারাগ্রাফে | "সমস্যা, আমাদের প্রস্তাবিত method, benchmark আর চারটা ফলাফল — ইচ্ছে করেই কম সংখ্যা দিয়ে।" |
| **I. Introduction** | প্রেরণা, ফাঁক, তিনটা contribution | "ফাঁকটা কোনো অনুপস্থিত algorithm-এর না, একটা নিয়ন্ত্রিত তুলনার।" |
| **II. Related Work** | SHAP ও LIME কী, explanation কীভাবে মূল্যায়ন হয়, আগের তুলনায় কী বাদ, আর fusion নিয়ে কী জানা আছে | "Aggregation-এর ধারণার কৃতিত্ব আমরা দাবি করি না; উদ্ধৃত করি আর দুটো published rule-কে baseline হিসেবে ব্যবহার করি।" |
| **III-A. Datasets, models, protocol** | তিনটা dataset, ছয়টা model, দুই-track নকশা | "Track A tune ও score করে; Track B এমন একটা model ব্যাখ্যা করে যার test partition tuning-এ অংশ নেয়নি।" |
| **III-B. Explainers and metrics** | backend, আর প্রতিটা metric-এর সংজ্ঞা | "প্রতিটা metric প্রকাশিত literature থেকে নেওয়া ও উদ্ধৃত — নতুনত্ব নকশায়, মাপকাঠিতে না।" |
| **III-C. The hybrid consensus method** | Algorithm 1 আর fidelity-weighted rule | "দুটো rule baseline, তৃতীয়টা আমাদের, আর তার weight test instance থেকে আলাদা একটা split-এ fit করা।" |
| **III-D. Four traps** | পদ্ধতিগত ফাঁদ ও প্রত্যেকটা কীভাবে বন্ধ করা হলো | "ভুল হলে এগুলোর একটাও output-এ দেখা যায় না — এজন্যই এগুলো পেপারে আছে।" |
| **III-E. Statistics** | pre-registered statistical plan | "Non-parametric, family-র ভিতরে Holm correction, প্রতিটা test-এ effect size, প্রতিটা estimate-এ interval।" |
| **IV. Results** | প্রতি research question-এ একটা subsection, সাথে hypothesis decision table | "Table I-এর প্রতিটা সারি কোনো ফলাফলের আগে commit করা protocol-এর বিরুদ্ধে যাচাইযোগ্য।" |
| **V. Discussion** | ফলাফলগুলোর ব্যবহারিক মানে | "দুটো explanation-এর intersection পড়া কোনো robustness check না।" |
| **VI. Limitations** | validity-র ঝুঁকি, additivity gate ব্যর্থতা সহ | "পাঠককে তোলার আগেই আমরা এগুলো বলে দিই।" |
| **VII. Conclusion** | চারটা ফলাফল পাঁচ বাক্যে | — |
| **Ethics and Reproducibility** | কোনো human subject নেই, কোনো clinical দাবি নেই, আর সংখ্যাগুলো কীভাবে তৈরি | "রোগগুলো একটা computer-science প্রশ্নের benchmark।" |
| **Author Contributions** | কে কী করেছে | পার্ট ১৩ দেখো |
| **Figures** | F4 agreement heat map, F13 fusion-এর ফলাফল | "F13-ই negative ফলাফল: প্রতিটা cell শূন্যের বাঁ দিকে।" |
| **Table I** | এগারোটা pre-registered hypothesis আর তাদের সিদ্ধান্ত | "একটা টেবিল, এগারোটা সিদ্ধান্ত, সব pre-registered।" |

দীর্ঘ report (`report/main.pdf`) একই study-র পূর্ণ রূপ: ৬ পাতার পেপারে যে table ও figure রাখা
যায়নি সেগুলো এতে আছে, সাথে চারটা appendix (পূর্ণ ফলাফল, citation সহ reference feature set,
environment-এর রেকর্ড, আর repository-র গাইড)।

---

## ১২. স্যারের সম্ভাব্য প্রশ্ন ও উত্তর

ছোট উত্তর, যেগুলো সত্যিই মুখে বলা যায়। পড়ে শোনাবে না — জেনে রাখবে।

### কাজটা কী, সে বিষয়ে

**"তোমরা আসলে কী করেছ?"**
> আমরা SHAP আর LIME-কে একটাই নিয়ন্ত্রিত protocol-এ তুলনা করেছি — ছয়টা classifier, তিনটা মেডিকেল
> dataset, আঠারোটা combination, ১,৭৪৬টা ব্যাখ্যা করা prediction — agreement, তিন ধরনের
> stability, faithfulness, খরচ আর plausibility-র উপর। আর দুটোকে মেশানোর একটা method প্রস্তাব
> করেছি, fidelity-weighted rank fusion, যেটা দুটো published fusion baseline-এর বিরুদ্ধে পরীক্ষা
> করা। তুলনাটা স্পষ্ট উত্তর দিয়েছে; মেশানোর পদ্ধতিটা তার মানদণ্ড পার করেনি, আর কেন করেনি সেটা
> আমরা ব্যাখ্যা করেছি।

**"এখানে নতুন কী? তোমাদের নিজের কী?"**
> তিনটা জিনিস। Benchmark-এর নকশা — model family, dataset আর explainer একসাথে বদলানো, একটাই
> leakage-safe protocol-এ, আগেই লেখা hypothesis নিয়ে। Stability-কে explainer randomness, input
> perturbation আর retraining variance-এ ভাগ করা। আর fidelity-weighted fusion rule, তার
> leakage-safe calibration split সহ। Metric-গুলো সব প্রকাশিত literature থেকে ও উদ্ধৃত;
> explanation মেশানোর ধারণাটাও আগের কাজ, সেটাও আমরা উদ্ধৃত করেছি।

**"Explanation fusion তোমরা আবিষ্কার করেছ?"**
> না, আর পেপারেই সেটা বলা আছে। Mean-rank fusion আর reciprocal-rank fusion প্রকাশিত; আমরা
> সেগুলোকে baseline হিসেবে ব্যবহার করেছি। আমাদের নিজের হলো — দুই explainer-কে সমানভাবে না ধরে
> তাদের **মাপা** fidelity অনুযায়ী weight দেওয়া, সেই weight একটা আলাদা calibration split-এ fit
> করা যাতে rule নিজের test set-এ tune না হয়, আর দুজনের গড়ের বদলে ভালোটার বিরুদ্ধে পরীক্ষা করা।

### Hybrid আর টাইটেল নিয়ে

**"তোমাদের method টাইটেলে আছে কিন্তু কাজ করেনি। এটা contribution কীভাবে?"**
> কারণ আমরা মানদণ্ডটা সৎভাবে ঠিক করেছি আর যা পেয়েছি তাই বলেছি। মানদণ্ড ছিল একই cell-এ SHAP আর
> LIME-এর **ভালোটাকে** হারানো — তাদের গড়কে না, যেটা যেকোনো কিছুই হারাতে পারত। Null ফলাফলটা
> কিছু চালানোর আগেই বৈধ ফলাফল হিসেবে লেখা ছিল, তাই আমরা চুপচাপ সফলতার সংজ্ঞা বদলাতে পারতাম না।
> আর কেন হলো না তার কারণ আমরা বলতে পারি: fusion-এর দরকার পরিপূরক input, আর এই দুটো পরিপূরক না —
> এদের মিল যথেষ্ট বেশি এবং একজন ধারাবাহিকভাবে ভালো, তাই গড় করলে শক্তিশালী signal-টাই পাতলা হয়।
> ভাগ্যক্রমে একটা positive ফলাফল পেয়ে কারণ ব্যাখ্যা করতে না পারা পেপার এর চেয়ে কম মূল্যবান হতো।

**"তাহলে টাইটেলে রাখলে কেন?"**
> কারণ এটা তিনটা contribution-এর একটা, আর পাঠকের জানার অধিকার আছে যে পেপারটা শুধু তুলনা করছে না,
> একটা method-ও প্রস্তাব করছে। "And Its Limits" ওই দাবির সৎ অর্ধেক — আমরা সীমা নির্ধারণ করেছি
> কখন explanation fusion চেষ্টা করার মতো, আর সেটা একটা ফলাফল।

**"Fusion কাজ করাতে কী লাগত?"**
> সত্যিকারের পরিপূরক base explainer — যেমন একটা gradient-ভিত্তিক আর একটা perturbation-ভিত্তিক
> method, বা এমন দুটো explainer যাদের গুণমান কাছাকাছি, একজন আরেকজনের উপর প্রভাবশালী না। এটাই
> স্বাভাবিক পরবর্তী কাজ, আর future work-এ লেখা আছে।

### ফলাফল নিয়ে

**"ρ = 0.74 — এটা ভালো মিল না খারাপ?"**
> যথেষ্ট ভালো, আর disagreement-এর literature যা ধারণা দেয় তার চেয়ে বেশি। কিন্তু top-5 তালিকা
> median-এ একটা feature-এ আলাদা হয়, তাই দুটোকে একটার বদলে আরেকটা ধরে নেওয়ার মতো যথেষ্ট না।
> বাক্যের দুই অর্ধেকই ফলাফল।

**"তাহলে যে feature-গুলোতে দুজন একমত, সেগুলোই নিলে হয় না?"**
> আমাদের ফলাফল ঠিক এই কাজটার বিরুদ্ধেই সতর্ক করে। Intersection নিলে প্রত্যেক তালিকা থেকে প্রায়
> একটা feature চুপচাপ বাদ যায়, আর কোনটা বাদ গেল তা জানার উপায় থাকে না। দেখতে robustness check
> লাগে, কিন্তু সেটা না।

**"তাহলে সবসময় SHAP ব্যবহার করা উচিত?"**
> যেখানে tree backend আছে, এই প্রমাণের ভিত্তিতে হ্যাঁ — বেশি faithful, প্রায় ২৫ গুণ সস্তা, আর
> input-এর ছোট পরিবর্তনে অনেক বেশি robust। LIME-এর পক্ষে বাকি থাকে যে সে যেকোনো model-এ কাজ করে,
> আর আমাদের stacking cell-গুলো দেখায় সেটা সত্যিকারের সুবিধা।

**"SHAP তো deterministic, তাহলে তার stability মাপছ কেন?"**
> কারণ determinism আর robustness আলাদা ধর্ম, আর এটাই আমাদের তিনটা contribution-এর একটা। Tree
> SHAP re-run-এ বিট-টু-বিট একই সংখ্যা দেয় — সেটা আমরা আগেই method-এর ধর্ম হিসেবে ঘোষণা করেছি।
> কিন্তু তার cross-refit agreement 0.83: একই data-র resample-এ একই model আবার fit করলে global
> ranking নড়ে। অস্থিরতাটা model-এ, explainer-এ না, আর determinism সেটা সরাতে পারে না।

**"সবচেয়ে সহজ dataset-এ মিল সবচেয়ে কম কেন? এটা significant?"**
> এটা statistically প্রতিষ্ঠিত না — H1c reject হয়নি, আর তিনটা dataset দিয়ে trend পরীক্ষার power
> নেই। পর্যবেক্ষণ হিসেবে: সহজ task-এ মিল 0.63, আর কঠিন দুটোতে 0.76 ও 0.78। আমাদের প্রস্তাবিত
> ব্যাখ্যা কঠিনতা না, feature redundancy: একাধিক feature একা একাই যথেষ্ট হলে দুটো explainer
> দুজনেই স্থানীয়ভাবে faithful থেকেও আলাদা নাম বলতে পারে।

**"AOPC কী, এক বাক্যে?"**
> একটা explanation যে feature-গুলোকে আগে রেখেছে সেগুলো মুছে training median বসাও, আর দেখো
> model-এর নিজের predicted probability কতটা পড়ে; সর্বোচ্চ প্রথম দশটা মোছার গড় নাও (feature কম
> হলে সবগুলো)। Faithful ranking হলে এটা দ্রুত পড়ে।

**"Random-ranking control কেন দরকার?"**
> কারণ যেকোনো কয়েকটা feature মুছলেই prediction নড়ে, তাই একা AOPC-র কোনো scale নেই। Control পায়
> 0.075; SHAP পায় 0.260। Control ছাড়া দ্বিতীয় সংখ্যাটার কোনো মানে নেই।

### পদ্ধতি ও কঠোরতা নিয়ে

**"তোমরা ফলাফল দেখে কিছু বদলাও নি, সেটা আমি কীভাবে জানব?"**
> এগারোটা hypothesis, metric-এর সূত্র, test আর সীমা সহ protocol-টা কোনো ফলাফল-সহ commit হওয়ার
> আগেই নিজের একটা commit-এ git-এ জমা দেওয়া। History-ই প্রমাণ। পরের প্রতিটা পরিবর্তন তারিখ ও কারণ
> সহ amendment log-এ। আর একটা gate ফেল করেছে — PIMA/LightGBM-এর additivity check — যেটা আমরা
> ঢিলা না করে রিপোর্ট করেছি, কারণ ফেল করতে দেখে gate ঢিলা করা ঠিক সেই জিনিস যেটা
> pre-registration আটকাতে আছে।

**"মাত্র তিনটা dataset কেন?"**
> সৎ উত্তর: একটা undergraduate থিসিসের compute আর পরিসর। আমরা তিনটা একরকম dataset না নিয়ে
> কঠিনতার একটা ধারা জুড়ে তিনটা নিয়েছি, যাতে নকশা দিয়ে প্রশস্ততার কিছুটা ঘাটতি পূরণ হয়।
> আমাদের limitations-এর তালিকায় এটা সবার আগে আছে — dataset-জুড়ে সাধারণীকরণ তিনটা বিন্দুর উপর
> দাঁড়ানো।

**"কোনো মেডিকেল দাবি না করলে মেডিকেল data কেন?"**
> কারণ এগুলোই ছোট public tabular benchmark হিসেবে standard, আর ব্যবহারিকভাবে explanation
> এখানেই সবচেয়ে বেশি দরকার। রোগগুলো একটা computer-science প্রশ্নের benchmark; এখানকার কোনো
> model রোগনির্ণয়ের উপযোগী না, আর ethics statement-এ সেটা লেখা আছে।

**"Deep learning কেন নেই?"**
> Tabular data-তে সবচেয়ে ভালো কাজ করে tree ensemble, আর post-hoc explanation-এর দরকার এই
> model-গুলোরই। একটা neural network যোগ করলে একটা axis বাড়ত, কিন্তু আমাদের প্রশ্নের উত্তর
> এগোত না।

**"SMOTE / oversampling কেন করলে না?"**
> Imbalance মৃদু, আর synthetic oversampling সেই data manifold বদলে দেয় যেখান থেকে LIME নমুনা
> নেয় আর SHAP তার background নেয়। ঠিক যে তুলনাটা করছি, সেটাই দূষিত হতো। সিদ্ধান্তটা protocol-এ
> লেখা, যাতে এটা একটা সিদ্ধান্ত হিসেবেই পড়া হয়, ভুলে বাদ পড়া হিসেবে না।

**"পেপারের সংখ্যাগুলো কোথা থেকে আসে?"**
> সব কোড থেকে। Pipeline `results/` লেখে, একটা script সেগুলো থেকে `tables/findings.tex`-এ LaTeX
> macro বানায়, আর পেপার সেই macro ব্যবহার করে — তাই কোনো সংখ্যা হাতে টাইপ করা না। `make all`
> দিলে raw data থেকে PDF পর্যন্ত পুরোটা আবার তৈরি হয়।

**"এটা একটা class project থেকে আলাদা কীভাবে?"**
> চারটা পদ্ধতিগত ফাঁদ, যেগুলো এই ধরনের তুলনাকে চুপচাপ অবৈধ করে দেয়, আর যেগুলো আমরা খুঁজে বের
> করে বন্ধ করেছি এবং লিখে রেখেছি: দুই explainer-কে একই রাশি ব্যাখ্যা করতে হবে আর আমরা সেটা যাচাই
> করি; SHAP-এর background আসল row হতে হবে, k-means centroid না; ranking-এর আগে one-hot column
> আবার feature-এ যোগ করতে হবে; আর control ছাড়া fidelity-র কোনো scale নেই। ভুল হলে চারটার
> একটাও output-এ দেখা যায় না।

---

## ১৩. কে কোন অংশের উত্তর দেবে

প্রত্যেকের নিজের অংশের প্রশ্ন কারো সাহায্য ছাড়া উত্তর দিতে পারা দরকার। এটা
`docs/CONTRIBUTIONS.md` অনুযায়ী — আসল ভাগ আলাদা হলে ওই ফাইলটা ঠিক করে নাও।

| লেখক | যার দায়িত্ব | যেসব প্রশ্নের জন্য প্রস্তুত থাকবে |
|---|---|---|
| **Samira Oasela Antika** | literature survey, আগের তুলনার সাপেক্ষে অবস্থান, freeze করা reference feature set | "আগের তুলনাগুলো কী করেছে?" · "Reference feature কোথা থেকে এল?" · "RQ5 কেবল সমর্থক প্রমাণ কেন?" |
| **Yeasmin Akter Shuchona** | dataset, উৎস, preprocessing, performance track | "এই তিনটা dataset কেন?" · "PIMA-র অসম্ভব শূন্যগুলো নিয়ে কী করেছ?" · "Hyperparameter কীভাবে বাছা হলো?" · "SMOTE কেন না?" |
| **MD. Marajul Haque** | explanation pipeline, metric suite, statistics, figure ও table তৈরির চেইন | "Fidelity-weighted rule কীভাবে কাজ করে?" · "AOPC কী?" · "Holm correction কেন?" · "Run-গুলো reproducible, কীভাবে জানো?" · চারটা ফাঁদ |
| **Gian Juti Tripura** | পেপার, report ও slides, limitations ও ethics | "সীমাবদ্ধতা কী কী?" · "কী দাবি তোমরা করছ না?" · "Ethics-এর অবস্থান কী?" |

সবার জানা থাকা দরকার: পার্ট ১ (তিনটা contribution), পার্ট ৭ (ফলাফল), আর "ব্যর্থ একটা method
contribution কীভাবে" — এই প্রশ্নের উত্তর।

---

## ১৪. মুখস্থ রাখার মতো সংখ্যা

| কী | মান |
|---|---|
| Cell (dataset × model) | **১৮** |
| ব্যাখ্যা করা prediction | **১,৭৪৬** |
| Pre-registered hypothesis | **১১** |
| Median Spearman ρ (SHAP vs LIME) | **0.74** |
| Median top-5 Jaccard | **0.67** (৫টার ৪টা feature মেলে) |
| Dataset অনুযায়ী ρ (PIMA / Heart / CKD) | 0.76 / 0.78 / **0.63** |
| Dataset অনুযায়ী ROC-AUC | 0.83 / 0.90 / 1.00 |
| AOPC: SHAP / LIME / random / best fusion | **0.260** / 0.224 / 0.075 / 0.248 |
| Local Lipschitz (কম ভালো): SHAP / LIME | **0.18** / 0.65 |
| Re-run CV: SHAP (tree) / LIME | **0** / 0.68 |
| Cross-refit agreement: SHAP / LIME | 0.83 / 0.76 |
| প্রতি explanation-এ সেকেন্ড: SHAP / LIME | 0.0019 / 0.0462 → **≈ 24.8× দ্রুত** |
| যেসব cell-এ fusion ভালোটাকে হারিয়েছে | **১৮-র মধ্যে ০** (median effect −0.80) |
| ρ > 0.50 মানদণ্ড পার করা cell | ১৮-র মধ্যে ১৫ |
| Explainer random control-কে হারিয়েছে | ৩৬-র মধ্যে ৩৬ তুলনায় |

---

## ১৫. শব্দকোষ

- **AOPC** — Area Over the Perturbation Curve। top-1, top-2, … top-10 feature মুছতে থাকলে
  model-এর predicted probability-র গড় পতন। বেশি = বেশি faithful।
- **Additivity check** — যাচাই করা যে base value + সব attribution-এর যোগফল model-এর আসল
  predicted probability-র সমান। ভিন্ন output space তুলনা করার বিরুদ্ধে আমাদের রক্ষাকবচ।
- **BCa bootstrap** — bias-corrected bootstrap confidence interval। একবার এটা একটা non-smooth
  statistic-এ ভুলভাবে প্রয়োগ হয়েছিল; protocol-এর একটা test ধরেছে।
- **Bootstrap resample** — training data থেকে প্রতিস্থাপন সহ নেওয়া নমুনা, যা দিয়ে model আবার fit
  করে দেখা হয় explanation টেকে কিনা।
- **Calibration split** — ব্যাখ্যা করা instance-এর ৩০%, যা দিয়ে fusion weight fit করা হয়; বাকি
  ৭০%-এ RQ6 পরীক্ষা করা হয়, দুটো মেশে না।
- **Cell** — একটা (dataset, model) জোড়া। মোট ১৮টা।
- **Coefficient of variation (CV)** — standard deviation ভাগ mean; re-run-এ explainer-এর সংখ্যা
  কতটা নড়ে।
- **Comprehensiveness / sufficiency** — ERASER-ধাঁচের fidelity মাপ, AOPC-র পাশাপাশি রিপোর্ট করা।
- **Effect size (rank-biserial)** — পার্থক্যটা কত বড়, sample size নির্বিশেষে। প্রতি cell-এ শত শত
  instance থাকলে এটা p-value-র চেয়ে বেশি গুরুত্বপূর্ণ।
- **Fidelity** — explanation model-এর আসল কাজের সাথে কতটা মেলে। অনুমান না, মাপা।
- **Holm correction** — একসাথে অনেক hypothesis পরীক্ষার সমন্বয়, যাতে আঠারোটা test চালিয়ে
  মিথ্যা positive তৈরি না হয়।
- **Jaccard@k** — দুটো top-k সেটের overlap: intersection-এর আকার ভাগ union-এর আকার।
- **Leakage** — যখন test data-র তথ্য training বা tuning-কে প্রভাবিত করে, ফলে ফলাফল বাস্তবের চেয়ে
  ভালো দেখায়। আমাদের দুই-track নকশা একটা নির্দিষ্ট ধরনের leakage আটকাতে আছে।
- **LIME** — Local Interpretable Model-agnostic Explanations: একটা prediction-এর আশেপাশে সরল
  linear model ফিট করে।
- **Local Lipschitz estimate** — ছোট একটা প্রতিবেশে input-এর প্রতি একক পরিবর্তনে explanation-এর
  সর্বোচ্চ পরিবর্তন। কম = বেশি robust।
- **Log-odds বনাম probability space** — model-এর output attribute করার দুটো আলাদা scale। দুটো
  মিশিয়ে ফেললে তুলনা চুপচাপ অবৈধ হয়ে যায় (ফাঁদ ১)।
- **Nested cross-validation** — বাইরের loop সৎ performance হিসাবের জন্য, ভিতরের loop
  hyperparameter বাছার জন্য। যে data-তে score করছ, তাতে tune করা আটকায়।
- **Off-manifold** — এমন input যা বাস্তব data-তৈরির প্রক্রিয়া কখনো তৈরি করত না, যেমন একটা
  feature-এ median বসিয়ে বানানো row।
- **Pre-registration** — কোনো ফলাফল দেখার আগেই hypothesis ও analysis-এর সিদ্ধান্ত লিখে ও commit
  করে রাখা।
- **RBO (rank-biased overlap)** — দুটো সাজানো তালিকার উপরের দিকে বেশি ওজন দেওয়া সাদৃশ্য, cut-off
  বাছতে হয় না। এর একটা floor আছে যা feature সংখ্যার উপর নির্ভর করে।
- **RRF (reciprocal-rank fusion)** — একটা published fusion baseline: প্রতিটা তালিকা দেয়
  1/(60 + rank)।
- **SHAP** — SHapley Additive exPlanations: cooperative game theory-র Shapley value দিয়ে
  prediction-কে feature-দের মধ্যে ভাগ করে।
- **Spearman ρ / Kendall τ** — দুটো ranking-এর মধ্যে correlation; 1 মানে হুবহু একই ক্রম।
- **Stacking ensemble** — একটা meta-learner দিয়ে কয়েকটা base model মেলানো model। এই study-তে
  এটাই একমাত্র model যার tree-নির্দিষ্ট SHAP backend নেই।
- **TreeSHAP / TreeExplainer** — tree model-এর জন্য হুবহু ও দ্রুত SHAP algorithm। Deterministic।
- **Wilcoxon signed-rank test** — একটা non-parametric paired test; সর্বত্র আমাদের default।

---

*ENG 672 থিসিস টিমের জন্য তৈরি। সংখ্যাগুলোর উৎস `tables/findings.tex`; এই ফাইল আর পেপারের মধ্যে
কোনো অমিল হলে পেপারটাই ঠিক, এই ফাইলটা পুরোনো।*
