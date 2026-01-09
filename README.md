# 🧠 Aura Vision – Real‑Time Emotion Recognition Web App

Aura Vision คือเว็บแอปพลิเคชันสำหรับ **ตรวจจับและวิเคราะห์อารมณ์จากใบหน้าแบบเรียลไทม์** ผ่านกล้องของผู้ใช้ โดยทำงานทั้งหมดบนฝั่งเบราว์เซอร์ (Client‑side) เพื่อความเป็นส่วนตัวและความรวดเร็ว

โปรเจกต์นี้พัฒนาขึ้นเพื่อการศึกษา โดยผสานเทคโนโลยีด้าน **Computer Vision** และ **Machine Learning** เข้ากับ Web Technology สมัยใหม่

---

## ✨ คุณสมบัติหลัก (Features)

* 🎥 ตรวจจับใบหน้าจากกล้องแบบเรียลไทม์
* 🧠 วิเคราะห์อารมณ์จากใบหน้าโดยใช้โมเดล **ONNX**
* ⚡ ประมวลผลด้วย **WebAssembly (WASM)** บน Browser
* 🔒 ไม่ส่งข้อมูลภาพออกไปภายนอก (Privacy‑friendly)
* 🎨 UI แบบ Cinematic / Luxury พร้อมธีมตามอารมณ์

---

## 😄 อารมณ์ที่รองรับ (Supported Emotions)

ระบบรองรับการจำแนกอารมณ์ทั้งหมด **7 ประเภท** ตามโมเดลที่ใช้งาน:

* 😠 **Angry** – ความโกรธ
* 🤢 **Disgust** – ความรังเกียจ
* 😨 **Fear** – ความกลัว
* 😀 **Happy** – ความสุข
* 😐 **Neutral** – ปกติ
* 😢 **Sad** – ความเศร้า
* 😲 **Surprise** – ความประหลาดใจ

แต่ละอารมณ์จะแสดงผลด้วย **สี, gradient และ animation** ที่แตกต่างกัน เพื่อสื่ออารมณ์ได้ชัดเจน

---

## 🛠 เทคโนโลยีที่ใช้ (Tech Stack)

| หมวด             | เทคโนโลยี          |
| ---------------- | ------------------ |
| Frontend         | Next.js (React)    |
| Computer Vision  | OpenCV.js          |
| Machine Learning | ONNX Runtime Web   |
| Runtime          | WebAssembly (WASM) |
| Styling          | Tailwind CSS       |
| Deployment       | Vercel             |

---

## ⚙️ หลักการทำงานของระบบ (System Workflow)

1. เปิดกล้องผ่าน Browser (WebRTC)
2. ใช้ **OpenCV.js** ตรวจจับใบหน้า (Haar Cascade)
3. ตัดภาพใบหน้าและทำการ Preprocess
4. ส่งข้อมูลเข้าโมเดล **ONNX Emotion Classification**
5. คำนวณผลลัพธ์และแสดงอารมณ์ + Confidence แบบเรียลไทม์

---

## 📁 โครงสร้างไฟล์สำคัญ (Project Structure)

```text
/public
 ├─ /opencv
 │   ├─ opencv.js
 │   └─ haarcascade_frontalface_default.xml
 ├─ /onnx
 │   └─ *.wasm
 └─ /models
     ├─ emotion_yolo11n_cls.onnx
     └─ classes.json

/app
 └─ page.tsx   // logic หลักของระบบ
```

---

## 🚀 การติดตั้งและใช้งาน (Installation & Run)

```bash
npm install
npm run dev
```

เปิด Browser ที่

```
http://localhost:3000
```

> ⚠️ แนะนำให้ใช้ **Chrome / Edge** เพื่อรองรับ WASM ได้ดีที่สุด

---

## 🔐 ความเป็นส่วนตัว (Privacy)

* ภาพจากกล้อง **ไม่ถูกส่งไปยัง Server**
* การประมวลผลทั้งหมดทำงานบนเครื่องผู้ใช้
* ไม่มีการบันทึกภาพหรือข้อมูลอารมณ์

---

## 👥 สมาชิกกลุ่มผู้พัฒนา (Team Members)

| รหัสนักศึกษา | ชื่อ‑นามสกุล           |
| ------------ | ---------------------- |
| 67022535     | นายการัญยภาส กันทะเนตร |
| 67022748     | นายพีรพัฒน์ แสวงรัมย์  |
| 67023031     | นางสาวอรชพร กลิ่นชื่น  |

---

## 📌 หมายเหตุ

โปรเจกต์นี้จัดทำขึ้นเพื่อการศึกษาและการเรียนรู้เทคโนโลยีด้าน
**AI, Computer Vision และ Web Application Development**

---

> *“Emotion is the silent language of the soul — Aura Vision makes it visible.”* ✨
