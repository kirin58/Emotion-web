"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

/* ================= THEME ================= */

const getEmotionTheme = (emotion: string) => {
  const e = (emotion || "").toLowerCase();

  switch (e) {
    case "happy":
      return {
        label: "happy",
        sub: "Positive Valence",
        gradient: "from-emerald-400 via-teal-300 to-cyan-300",
        shadow: "shadow-emerald-500/40",
        hex: "#34d399",
      };

    case "sad":
      return {
        label: "sad",
        sub: "Negative Valence",
        gradient: "from-blue-400 via-indigo-300 to-violet-400",
        shadow: "shadow-blue-500/40",
        hex: "#60a5fa",
      };

    case "angry":
      return {
        label: "angry",
        sub: "High Arousal",
        gradient: "from-red-500 via-rose-400 to-orange-400",
        shadow: "shadow-red-500/40",
        hex: "#f43f5e",
      };

    case "surprise":
      return {
        label: "surprise",
        sub: "Sudden Stimuli",
        gradient: "from-amber-300 via-yellow-300 to-orange-300",
        shadow: "shadow-amber-400/40",
        hex: "#fbbf24",
      };

    case "neutral":
      return {
        label: "neutral",
        sub: "Baseline State",
        gradient: "from-slate-200 via-gray-300 to-zinc-400",
        shadow: "shadow-white/20",
        hex: "#e2e8f0",
      };

    /* ====== เพิ่มใหม่ ====== */

    case "fear":
      return {
        label: "fear",
        sub: "Threat Response",
        gradient: "from-purple-500 via-indigo-500 to-blue-600",
        shadow: "shadow-purple-500/40",
        hex: "#8b5cf6",
      };

    case "disgust":
      return {
        label: "disgust",
        sub: "Aversive Reaction",
        gradient: "from-lime-500 via-green-500 to-emerald-600",
        shadow: "shadow-lime-500/40",
        hex: "#84cc16",
      };

    default:
      return {
        label: "Scanning",
        sub: "Awaiting Input",
        gradient: "from-cyan-400 via-blue-400 to-indigo-400",
        shadow: "shadow-cyan-500/30",
        hex: "#22d3ee",
      };
  }
};

type CvType = any;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);
  const loopIdRef = useRef<number>(0);

  const isInferringRef = useRef(false);
  const lastInferTimeRef = useRef(0);
  const INFER_INTERVAL = 150;

  const [initStatus, setInitStatus] = useState("System Initializing...");
  const [isLoading, setIsLoading] = useState(true);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [emotionData, setEmotionData] = useState<{ label: string; conf: number } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  /* ================= OpenCV ================= */

  async function loadOpenCV() {
    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const s = document.createElement("script");
      s.src = "/opencv/opencv.js";
      s.async = true;
      s.onload = () => {
        const cv = (window as any).cv;
        if (!cv) return reject();
        cv.onRuntimeInitialized = () => {
          cvRef.current = cv;
          resolve();
        };
      };
      s.onerror = reject;
      document.body.appendChild(s);
    });
  }

  async function loadCascade() {
    const cv = cvRef.current;
    const path = "haarcascade_frontalface_default.xml";
    try {
      cv.FS_stat(path);
    } catch {
      const res = await fetch("/opencv/haarcascade_frontalface_default.xml");
      const data = new Uint8Array(await res.arrayBuffer());
      cv.FS_createDataFile("/", path, data, true, false, false);
    }
    const c = new cv.CascadeClassifier();
    c.load(path);
    faceCascadeRef.current = c;
  }

  /* ================= ONNX ================= */

  async function loadModel() {
    // 🔒 FIX สำหรับ Vercel + Browser
    ort.env.wasm.wasmPaths = "/onnx/";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = false;
    (ort.env.wasm as any).proxy = false;

    const session = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "disabled",
      }
    );

    sessionRef.current = session;
    const res = await fetch("/models/classes.json");
    classesRef.current = await res.json();
  }

  /* ================= CAMERA ================= */

  async function startCamera() {
    setErrorMsg(null);
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    videoRef.current!.srcObject = stream;
    await videoRef.current!.play();
    setIsCameraActive(true);
    loopIdRef.current = requestAnimationFrame(loop);
  }

  function stopCamera() {
    cancelAnimationFrame(loopIdRef.current);
    const stream = videoRef.current?.srcObject as MediaStream;
    stream?.getTracks().forEach(t => t.stop());
    setIsCameraActive(false);
    setEmotionData(null);
  }

  /* ================= INFERENCE ================= */

  function preprocess(face: HTMLCanvasElement) {
    const s = 64;
    const c = document.createElement("canvas");
    c.width = c.height = s;
    c.getContext("2d")!.drawImage(face, 0, 0, s, s);
    const d = c.getContext("2d")!.getImageData(0, 0, s, s).data;
    const f = new Float32Array(1 * 3 * s * s);
    let k = 0;
    for (let ch = 0; ch < 3; ch++)
      for (let i = 0; i < s * s; i++)
        f[k++] = d[i * 4 + ch] / 255;
    return new ort.Tensor("float32", f, [1, 3, s, s]);
  }

  async function runInference(rect: any) {
    if (isInferringRef.current) return;
    isInferringRef.current = true;

    try {
      const c = document.createElement("canvas");
      c.width = rect.width;
      c.height = rect.height;
      c.getContext("2d")!.drawImage(canvasRef.current!, rect.x, rect.y, rect.width, rect.height, 0, 0, rect.width, rect.height);

      const input = preprocess(c);
      const out = await sessionRef.current!.run({ [sessionRef.current!.inputNames[0]]: input });
      const logits = out[sessionRef.current!.outputNames[0]].data as Float32Array;

      let max = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[max]) max = i;

      setEmotionData({ label: classesRef.current![max], conf: logits[max] });
    } finally {
      isInferringRef.current = false;
    }
  }

  /* ================= LOOP ================= */

  function loop() {
    const cv = cvRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const faceCascade = faceCascadeRef.current;
    if (!cv || !video || !canvas || !faceCascade) return;

    const ctx = canvas.getContext("2d")!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const faces = new cv.RectVector();
    faceCascade.detectMultiScale(gray, faces, 1.2, 5);

    if (faces.size() > 0) {
      const r = faces.get(0);
      const now = performance.now();
      if (now - lastInferTimeRef.current > INFER_INTERVAL) {
        lastInferTimeRef.current = now;
        runInference(r);
      }
    }

    src.delete(); gray.delete(); faces.delete();
    loopIdRef.current = requestAnimationFrame(loop);
  }

  /* ================= INIT ================= */

  useEffect(() => {
    (async () => {
      try {
        await loadOpenCV();
        await loadCascade();
        await loadModel();
        setIsLoading(false);
        setInitStatus("System Ready");
      } catch (e: any) {
        setErrorMsg(e.message);
      }
    })();
  }, []);

  const currentTheme = getEmotionTheme(emotionData?.label ?? "");

  return (
    <main className="min-h-screen bg-[#050505] text-slate-200 flex items-center justify-center p-4 md:p-8 font-sans overflow-hidden relative selection:bg-white/20">
      {/* --- Ambient Background --- */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_at_50%_0%,#1a1a1a,transparent_70%)]"></div>
        <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[60vw] h-[60vw] rounded-full bg-gradient-to-r ${currentTheme.gradient} opacity-5 blur-[120px] transition-all duration-1000`}></div>
      </div>

      {/* --- Main Interface Container --- */}
      <div className="relative z-10 w-full max-w-6xl h-[80vh] md:h-[700px] flex flex-col md:flex-row bg-[#0a0a0a] rounded-[2rem] border border-white/5 shadow-[0_0_50px_-20px_rgba(0,0,0,0.7)] overflow-hidden ring-1 ring-white/5">
        {/* === LEFT: Video Viewport === */}
        <div className="relative flex-1 bg-black/40 h-full md:h-auto overflow-hidden group">
          {/* Header Overlay */}
          <div className="absolute top-0 left-0 right-0 p-6 flex justify-between items-start z-30 bg-gradient-to-b from-black/80 to-transparent">
            <div>
              <h1 className="text-lg font-bold text-white tracking-widest uppercase opacity-90">Aura Vision</h1>
              <div className="flex items-center gap-2 mt-1">
                <span className={`w-1.5 h-1.5 rounded-full ${isCameraActive ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-red-500'}`}></span>
                <span className="text-[10px] text-white/40 uppercase tracking-wider">{isCameraActive ? "Online" : "Offline"}</span>
              </div>
            </div>
          </div>

          {/* Video Canvas & Placeholder */}
          <div className="absolute inset-0 flex items-center justify-center">
            {!isCameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center z-20 p-8 text-center bg-black/60 backdrop-blur-sm">
                {isLoading && !errorMsg && (
                  <div className="flex flex-col items-center gap-4 animate-in fade-in duration-500">
                    <div className="w-12 h-12 rounded-full border-[3px] border-white/10 border-t-white/80 animate-spin"></div>
                    <p className="text-xs font-mono text-white/50 tracking-[0.2em] uppercase animate-pulse">{initStatus}</p>
                  </div>
                )}

                {!isLoading && !errorMsg && (
                  <div className="flex flex-col items-center gap-4 animate-in zoom-in duration-300">
                    <div className="w-16 h-16 rounded-full bg-white/5 border border-white/10 flex items-center justify-center backdrop-blur-md shadow-[0_0_30px_-10px_rgba(255,255,255,0.1)]">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor" className="w-6 h-6 text-white/70"><path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" /></svg>
                    </div>
                    <p className="text-xs font-mono text-white/40 tracking-widest uppercase">{initStatus}</p>
                  </div>
                )}

                {errorMsg && (
                  <div className="max-w-xs w-full p-6 bg-red-500/10 border border-red-500/20 rounded-2xl backdrop-blur-md animate-in fade-in slide-in-from-bottom-5">
                    <div className="flex flex-col items-center gap-3">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-8 h-8 text-red-400"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" /></svg>
                      <h3 className="text-red-200 font-medium text-sm">Initialization Failed</h3>
                      <p className="text-[10px] text-red-300/50 text-center font-mono">{errorMsg}</p>
                      <button onClick={() => window.location.reload()} className="mt-2 px-4 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-200 text-xs rounded-full transition-colors border border-red-500/20">Retry</button>
                    </div>
                  </div>
                )}
              </div>
            )}

            <video ref={videoRef} className="hidden" playsInline muted autoPlay />
            <canvas ref={canvasRef} className={`w-full h-full object-cover transition-opacity duration-700 ${isCameraActive ? 'opacity-100' : 'opacity-0'}`} />
          </div>

          <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:100px_100px] pointer-events-none opacity-20"></div>
        </div>

        {/* === RIGHT: Control Panel === */}
        <div className="w-full md:w-[360px] bg-[#0f0f0f] border-t md:border-t-0 md:border-l border-white/5 p-8 flex flex-col relative overflow-hidden">
          <div className={`absolute -top-[100px] -right-[100px] w-[300px] h-[300px] bg-gradient-to-b ${currentTheme.gradient} opacity-10 blur-[80px] rounded-full transition-colors duration-1000 pointer-events-none`}></div>

          <div className="flex-1 flex flex-col justify-center space-y-10 relative z-10">
            {/* 1. Emotion Display */}
            <div className="space-y-1">
              <p className="text-[10px] font-bold text-white/30 uppercase tracking-[0.2em]">Detected State</p>
              <div className="h-20 flex flex-col justify-center">
                <h2 className={`text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r ${currentTheme.gradient} transition-all duration-500 drop-shadow-lg`}>
                  {emotionData ? currentTheme.label : "Waiting"}
                </h2>
                <p className="text-xs text-white/40 font-medium tracking-wide mt-1 transition-all duration-500">
                  {emotionData ? currentTheme.sub : isCameraActive ? "Looking for a face..." : "Please start the session"}
                </p>
              </div>
            </div>

            {/* 2. Confidence Visualization */}
            <div className="space-y-4">
              <div className="flex justify-between items-end border-b border-white/5 pb-2">
                <span className="text-[10px] text-white/40 uppercase tracking-widest">Confidence</span>
                <span className="text-xl font-light text-white tabular-nums">
                  {emotionData ? (emotionData.conf * 100).toFixed(0) : "0"}
                  <span className="text-xs text-white/30 ml-1">%</span>
                </span>
              </div>
              <div className="flex gap-1 h-1.5 w-full">
                {[...Array(10)].map((_, i) => (
                  <div
                    key={i}
                    className={`flex-1 rounded-full transition-all duration-300 ${emotionData && (emotionData.conf * 10) > i ? `bg-gradient-to-r ${currentTheme.gradient} opacity-100` : 'bg-white/5'}`}
                  ></div>
                ))}
              </div>
            </div>

            {/* 3. Tech Stats */}
            <div className="grid grid-cols-2 gap-3 pt-4">
              <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.05] flex flex-col justify-between h-24 hover:bg-white/[0.05] transition-colors">
                <div className="text-white/20"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4"><path d="M12 15a3 3 0 100-6 3 3 0 000 6z" /><path fillRule="evenodd" d="M1.323 11.447C2.811 6.976 7.028 3.75 12.001 3.75c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113-1.487 4.471-5.705 7.697-10.677 7.697-4.97 0-9.186-3.223-10.675-7.69a1.762 1.762 0 010-1.113zM17.25 12a5.25 5.25 0 11-10.5 0 5.25 5.25 0 0110.5 0z" clipRule="evenodd" /></svg></div>
                <div>
                  <div className="text-sm font-bold text-white/80">1280p</div>
                  <div className="text-[9px] text-white/30 uppercase tracking-widest mt-0.5">Resolution</div>
                </div>
              </div>
              <div className="p-4 rounded-xl bg-white/[0.03] border border-white/[0.05] flex flex-col justify-between h-24 hover:bg-white/[0.05] transition-colors">
                <div className="text-white/20"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4"><path fillRule="evenodd" d="M3 6a3 3 0 013-3h12a3 3 0 013 3v12a3 3 0 01-3 3H6a3 3 0 01-3-3V6zm14.25 6a.75.75 0 01-.22.53l-2.25 2.25a.75.75 0 11-1.06-1.06L15.44 12l-1.72-1.72a.75.75 0 111.06-1.06l2.25 2.25c.141.14.22.331.22.53zm-10.28 0a.75.75 0 01.22-.53l2.25-2.25a.75.75 0 111.06 1.06L8.56 12l1.72 1.72a.75.75 0 11-1.06 1.06l-2.25-2.25a.75.75 0 01-.22-.53z" clipRule="evenodd" /></svg></div>
                <div>
                  <div className="text-sm font-bold text-white/80">WASM</div>
                  <div className="text-[9px] text-white/30 uppercase tracking-widest mt-0.5">Backend</div>
                </div>
              </div>
            </div>
          </div>

          {/* --- Action Buttons (Updated) --- */}
          <div className="mt-6">
            {!isCameraActive ? (
              <button
                disabled={isLoading}
                onClick={startCamera}
                className="w-full py-4 rounded-xl font-bold text-xs tracking-[0.2em] uppercase transition-all duration-300 bg-white text-black hover:scale-[1.02] shadow-[0_0_20px_rgba(255,255,255,0.2)] hover:shadow-[0_0_30px_rgba(255,255,255,0.4)] disabled:opacity-50"
              >
                {isLoading ? "System Loading..." : "Start Experience"}
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="w-full py-4 rounded-xl font-bold text-xs tracking-[0.2em] uppercase transition-all duration-300 bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 hover:text-red-300"
              >
                Terminate Session
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="absolute bottom-4 left-0 right-0 text-center">
        <p className="text-[9px] text-white/10 uppercase tracking-[0.3em] font-light mix-blend-plus-lighter">
          Powered by ONNX Runtime & OpenCV.js
        </p>
      </div>
    </main>
  );
}
