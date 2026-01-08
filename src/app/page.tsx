"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

// --- Theme Configuration (Cinematic Luxury) ---
const getEmotionTheme = (emotion: string) => {
  const e = emotion.toLowerCase();
  switch (e) {
    case "happy":
      return {
        label: "Radiance",
        sub: "Positive Valence",
        gradient: "from-emerald-400 via-teal-300 to-cyan-300",
        shadow: "shadow-emerald-500/40",
        hex: "#34d399",
      };
    case "sad":
      return {
        label: "Melancholy",
        sub: "Negative Valence",
        gradient: "from-blue-400 via-indigo-300 to-violet-400",
        shadow: "shadow-blue-500/40",
        hex: "#60a5fa",
      };
    case "angry":
      return {
        label: "Intensity",
        sub: "High Arousal",
        gradient: "from-red-500 via-rose-400 to-orange-400",
        shadow: "shadow-red-500/40",
        hex: "#f43f5e",
      };
    case "surprise":
      return {
        label: "Wonder",
        sub: "Sudden Stimuli",
        gradient: "from-amber-300 via-yellow-300 to-orange-300",
        shadow: "shadow-amber-400/40",
        hex: "#fbbf24",
      };
    case "neutral":
      return {
        label: "Serenity",
        sub: "Baseline State",
        gradient: "from-slate-200 via-gray-300 to-zinc-400",
        shadow: "shadow-white/20",
        hex: "#e2e8f0",
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
  // --- Refs & State ---
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);
  const loopIdRef = useRef<number>(0);

  const [initStatus, setInitStatus] = useState<string>("System Initializing...");
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [emotionData, setEmotionData] = useState<{ label: string; conf: number } | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // --- Logic Blocks ---
  async function loadOpenCV() {
    if (typeof window === "undefined") return;
    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;
      script.onload = () => {
        const cv = (window as any).cv;
        if (!cv) return reject(new Error("CV Load Failed"));
        const waitReady = () => {
          if ((window as any).cv?.Mat) {
            cvRef.current = (window as any).cv;
            resolve();
          } else setTimeout(waitReady, 50);
        };
        if ("onRuntimeInitialized" in cv) cv.onRuntimeInitialized = () => waitReady();
        else waitReady();
      };
      script.onerror = () => reject(new Error("Failed to load OpenCV Script"));
      document.body.appendChild(script);
    });
  }

  async function loadCascade() {
    const cv = cvRef.current;
    const cascadePath = "haarcascade_frontalface_default.xml";
    try {
        cv.FS_stat(cascadePath);
        const faceCascade = new cv.CascadeClassifier();
        faceCascade.load(cascadePath);
        faceCascadeRef.current = faceCascade;
    } catch {
        const res = await fetch("/opencv/haarcascade_frontalface_default.xml");
        if (!res.ok) throw new Error("Cascade Missing");
        const data = new Uint8Array(await res.arrayBuffer());
        cv.FS_createDataFile("/", cascadePath, data, true, false, false);
        const faceCascade = new cv.CascadeClassifier();
        faceCascade.load(cascadePath);
        faceCascadeRef.current = faceCascade;
    }
  }

  async function loadModel() {
    try {
        ort.env.wasm.wasmPaths = "/onnx/"; 
        ort.env.wasm.proxy = true; 
        ort.env.wasm.numThreads = 1; 

        const session = await ort.InferenceSession.create("/models/emotion_yolo11n_cls.onnx", { 
            executionProviders: ["wasm"],
        });
        
        sessionRef.current = session;
        const clsRes = await fetch("/models/classes.json");
        if (!clsRes.ok) throw new Error("Classes JSON Missing");
        classesRef.current = await clsRes.json();
    } catch (e) {
        console.error("Model loading error:", e);
        throw e;
    }
  }

  async function startCamera() {
    setErrorMsg(null);
    try {
      setInitStatus("Accessing Optics...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise((resolve) => { 
            videoRef.current!.onloadedmetadata = () => resolve(true); 
        });
        await videoRef.current.play();
        setIsCameraActive(true);
        loopIdRef.current = requestAnimationFrame(loop);
      }
    } catch (e: any) {
      setErrorMsg(e?.message ?? "Camera Access Denied");
      setInitStatus("Connection Failed");
    }
  }

  // --- NEW: Stop Camera Function ---
  function stopCamera() {
    if (loopIdRef.current) {
        cancelAnimationFrame(loopIdRef.current);
        loopIdRef.current = 0;
    }
    
    if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
    }
    
    setIsCameraActive(false);
    setEmotionData(null);
    setInitStatus("System Ready");
  }

  // --- Processing ---
  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size; tmp.height = size;
    const ctx = tmp.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);
    const imgData = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(1 * 3 * size * size);
    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) float[idx++] = imgData[i * 4 + c] / 255.0;
    }
    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  function softmax(logits: Float32Array) {
    let max = -Infinity; for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  async function loop() {
    const cv = cvRef.current;
    const faceCascade = faceCascadeRef.current;
    const session = sessionRef.current;
    const classes = classesRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!cv || !faceCascade || !session || !classes || !video || !canvas || video.paused || video.ended) {
      if (isCameraActive) loopIdRef.current = requestAnimationFrame(loop);
      return;
    }

    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    let src: any, gray: any, faces: any;
    try {
      src = cv.imread(canvas);
      gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      faces = new cv.RectVector();
      
      faceCascade.detectMultiScale(gray, faces, 1.2, 5, 0, new cv.Size(64, 64));

      let bestRect: any = null;
      let bestArea = 0;
      for (let i = 0; i < faces.size(); i++) {
        const r = faces.get(i);
        const area = r.width * r.height;
        if (area > bestArea) { bestArea = area; bestRect = r; }
      }

      if (bestRect) {
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = bestRect.width; faceCanvas.height = bestRect.height;
        faceCanvas.getContext("2d")!.drawImage(canvas, bestRect.x, bestRect.y, bestRect.width, bestRect.height, 0, 0, bestRect.width, bestRect.height);

        const input = preprocessToTensor(faceCanvas);
        const feeds = { [session.inputNames[0]]: input };
        const out = await session.run(feeds);
        const logits = out[session.outputNames[0]].data as Float32Array;
        const probs = softmax(logits);

        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i;

        const predLabel = classes[maxIdx] ?? "Unknown";
        const predConf = probs[maxIdx] ?? 0;
        setEmotionData({ label: predLabel, conf: predConf });

        const theme = getEmotionTheme(predLabel);
        const x = bestRect.x; const y = bestRect.y; const w = bestRect.width; const h = bestRect.height;
        
        ctx.strokeStyle = theme.hex;
        ctx.lineWidth = 3;
        ctx.shadowColor = theme.hex;
        ctx.shadowBlur = 15;
        
        const len = Math.min(w, h) / 4;
        ctx.beginPath();
        ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
        ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
        ctx.moveTo(x + w, y + h - len); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - len, y + h);
        ctx.moveTo(x + len, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - len);
        ctx.stroke();

        ctx.shadowBlur = 0; 
      } else {
        setEmotionData(null);
      }
    } catch (e) {
      console.warn("Loop error:", e);
    } finally {
      if (src) src.delete(); 
      if (gray) gray.delete(); 
      if (faces) faces.delete();
      if (isCameraActive) loopIdRef.current = requestAnimationFrame(loop);
    }
  }

  // Init
  useEffect(() => {
    (async () => {
      try {
        setIsLoading(true);
        setInitStatus("Loading Vision Core...");
        await loadOpenCV();
        await loadCascade();
        await loadModel();
        setInitStatus("System Ready");
        setIsLoading(false);
      } catch (e: any) {
        setIsLoading(false);
        setErrorMsg(e?.message ?? "Init Failed");
        console.error(e);
      }
    })();
    return () => { if (loopIdRef.current) cancelAnimationFrame(loopIdRef.current); };
  }, []);

  const currentTheme = emotionData ? getEmotionTheme(emotionData.label) : getEmotionTheme("-");

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
                                    <button onClick={()=>window.location.reload()} className="mt-2 px-4 py-1.5 bg-red-500/20 hover:bg-red-500/30 text-red-200 text-xs rounded-full transition-colors border border-red-500/20">Retry</button>
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