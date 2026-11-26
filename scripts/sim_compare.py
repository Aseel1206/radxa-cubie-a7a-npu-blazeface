import glob, numpy as np, cv2, tensorflow as tf
D = "/workspace/mp/face_det"
img = cv2.cvtColor(cv2.imread("/workspace/mp/test128/face.png"), cv2.COLOR_BGR2RGB).astype(np.float32)
x = ((img - 127.5) / 127.5)[None]
it = tf.lite.Interpreter(model_path=f"{D}/blaze_face_short_range.tflite"); it.allocate_tensors()
it.set_tensor(it.get_input_details()[0]["index"], x); it.invoke()
od = {d["name"]: it.get_tensor(d["index"]).ravel() for d in it.get_output_details()}
reg, clf = od["regressors"], od["classificators"]
sc = 1 / (1 + np.exp(-clf)); top = int(np.argmax(sc)); print(f"float TFLite: best score {sc[top]:.3f} at anchor {top}; #>0.5: {(sc > .5).sum()}")
for t in ("pcq", "i16"):
    out = {}
    for f in glob.glob(f"{D}/inf_{t}/iter_0_*.tensor"):
        if f.endswith(".qnt.tensor") or "input" in f: continue
        a = np.loadtxt(f, dtype=np.float32).ravel(); out[a.size] = a
    c = out.get(896); r = out.get(896 * 16)
    if c is None or r is None: print(t, "outputs not found:", [(f.split('/')[-1], ) for f in fs]); continue
    s2 = 1 / (1 + np.exp(-c)); t2 = int(np.argmax(s2))
    print(f"{t}: best score {s2[t2]:.3f} at anchor {t2}; #>0.5: {(s2 > .5).sum()}; corr(logits) {np.corrcoef(c, clf)[0, 1]:.4f}; corr(regr) {np.corrcoef(r, reg)[0, 1]:.4f}; max|dlogit| {np.abs(c - clf).max():.2f}")
