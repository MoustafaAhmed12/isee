// camera-detection.component.ts
import { CommonModule } from '@angular/common';
import {
  Component,
  OnInit,
  OnDestroy,
  ViewChild,
  ElementRef,
} from '@angular/core';
import { InferenceSession, Tensor, env } from 'onnxruntime-web';
import * as ort from 'onnxruntime-web';

interface Detection {
  class: string;
  confidence: number;
  bbox: number[];
}

@Component({
  selector: 'app-camera-detection',
  imports: [CommonModule],
  templateUrl: './camera-detection.component.html',
  styleUrl: './camera-detection.component.css',
})
export class CameraDetectionComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasElement') canvasElement!: ElementRef<HTMLCanvasElement>;

  // حالة التطبيق
  isCameraOn = false;
  isModelLoaded = false;
  isLoading = false;
  isDarkMode = false;
  lastDetectedObject = '';
  cameraFacingMode: 'user' | 'environment' = 'environment';

  // المتغيرات التقنية
  private stream: MediaStream | null = null;
  private session: InferenceSession | null = null;
  private detectionInterval: any = null;
  private lastSpokenObject = '';
  private lastSpokenTime = 0;
  private readonly SPEECH_COOLDOWN = 3000; // 3 ثواني

  // أسماء الفئات الخاصة بـ YOLOv8n (مثال)
  private classNames = [
    'شخص',
    'دراجة',
    'سيارة',
    'دراجة نارية',
    'طائرة',
    'حافلة',
    'قطار',
    'شاحنة',
    'قارب',
    'إشارة مرور',
    'صنبور إطفاء',
    'علامة توقف',
    'عداد ركن',
    'مقعد',
    'طائر',
    'قطة',
    'كلب',
    'حصان',
    'خروف',
    'بقرة',
    'فيل',
    'دب',
    'حمار وحشي',
    'زرافة',
    'حقيبة ظهر',
    'مظلة',
    'حقيبة يد',
    'ربطة عنق',
    'حقيبة سفر',
    'الطائر الطائر',
    'القرص الطائر',
    'لوح تزلج',
    'لوح تزلج على الماء',
    'كرة قدم',
    'طائرة ورقية',
    'مضرب بيسبول',
    'قفاز بيسبول',
    'مزلجة',
    'لوح تزلج على الثلج',
    'كرة طائرة',
    'مضرب تنس',
    'زجاجة',
    'كأس نبيذ',
    'كأس',
    'شوكة',
    'سكين',
    'ملعقة',
    'وعاء',
    'موزة',
    'تفاح',
    'ساندويتش',
    'برتقال',
    'بروكلي',
    'جزر',
    'هوت دوج',
    'بيتزا',
    'دونات',
    'كيك',
    'كرسي',
    'أريكة',
    'نبتة وعاء',
    'سرير',
    'طاولة طعام',
    'مرحاض',
    'تلفاز',
    'لابتوب',
    'فأرة',
    'جهاز تحكم',
    'لوحة مفاتيح',
    'هاتف خلوي',
    'مايكروويف',
    'فرن',
    'محمصة خبز',
    'حوض',
    'ثلاجة',
    'كتاب',
    'ساعة',
    'مزهرية',
    'مقص',
    'دب محشي',
    'مجفف شعر',
    'فرشاة أسنان',
  ];

  // async ngOnInit() {
  //     env.wasm.wasmPaths = 'assets/wasm/';
  //   this.loadThemePreference();
  //   await this.loadModel();
  // }

  ngOnDestroy() {
    this.stopCamera();
    if (this.detectionInterval) {
      clearInterval(this.detectionInterval);
    }
  }

  async ngOnInit() {
    await this.configureONNXRuntime();
    this.loadThemePreference();
    await this.loadModel();
  }

  private async loadModel() {
    this.isLoading = true;
    try {
      // المحاولة الأولى: استخدام WebGL
      this.session = await InferenceSession.create(
        'assets/models/yolov8n.onnx',
        {
          executionProviders: ['webgl'],
          graphOptimizationLevel: 'all',
        }
      );
      this.isModelLoaded = true;
      console.log('✅ تم تحميل النموذج بنجاح');
      console.log('📥 أسماء الإدخالات:', this.session.inputNames);
      console.log('📤 أسماء الإخراجات:', this.session.outputNames);
    } catch (webglError) {
      console.warn('WebGL غير متاح، جاري استخدام WASM:', webglError);

      // المحاولة الثانية: استخدام WASM
      try {
        this.session = await InferenceSession.create(
          'assets/models/yolov8n.onnx',
          {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
          }
        );
        this.isModelLoaded = true;
        console.log('✅ تم تحميل النموذج بنجاح');
        console.log('📥 أسماء الإدخالات:', this.session.inputNames);
        console.log('📤 أسماء الإخراجات:', this.session.outputNames);
        console.log('تم تحميل النموذج بنجاح باستخدام WASM');
      } catch (wasmError) {
        console.error('فشل تحميل النموذج مع كلا المزودين:', wasmError);
        this.showModelError();
      }
    } finally {
      this.isLoading = false;
    }
  }

  private async configureONNXRuntime() {
    env.wasm.wasmPaths = '/assets/wasm/';
    env.wasm.numThreads = 1;
    env.wasm.simd = true;
  }

  private showModelError() {
    alert(
      'تعذر تحميل نموذج التعرف على الأشياء. تأكد من:\n\n1. وجود ملف yolov8n.onnx في مجلد assets/models/\n2. دعم المتصفح لـ WebGL أو WASM\n3. اتصال الإنترنت لتحميل الملفات المطلوبة'
    );
  }

  // تبديل وضع الكاميرا
  async toggleCamera() {
    if (this.isCameraOn) {
      this.stopCamera();
    } else {
      await this.startCamera();
    }
  }

  // تبديل بين الكاميرا الأمامية والخلفية
  async switchCamera() {
    if (this.isCameraOn) {
      this.stopCamera();
      this.cameraFacingMode =
        this.cameraFacingMode === 'user' ? 'environment' : 'user';
      await this.startCamera();
    } else {
      this.cameraFacingMode =
        this.cameraFacingMode === 'user' ? 'environment' : 'user';
    }
  }

  // تشغيل الكاميرا
  private async startCamera() {
    try {
      this.isLoading = true;
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: this.cameraFacingMode,
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      this.videoElement.nativeElement.srcObject = this.stream;
      this.isCameraOn = true;

      // بدء التعرف على الأشياء
      this.startDetection();
    } catch (error) {
      console.error('خطأ في تشغيل الكاميرا:', error);
      alert('تعذر الوصول إلى الكاميرا. يرجى التحقق من الأذونات.');
    } finally {
      this.isLoading = false;
    }
  }

  // إيقاف الكاميرا
  private stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    this.isCameraOn = false;
    this.lastDetectedObject = '';

    if (this.detectionInterval) {
      clearInterval(this.detectionInterval);
      this.detectionInterval = null;
    }
  }

  // بدء التعرف على الأشياء
  private startDetection() {
    this.detectionInterval = setInterval(async () => {
      if (
        this.videoElement.nativeElement.readyState ===
        HTMLMediaElement.HAVE_ENOUGH_DATA
      ) {
        await this.detectObjects();
      }
    }, 2000); // كل ثانيتين
  }

  // التعرف على الأشياء
private async detectObjects() {
  console.log('بدء التعرف على الأشياء...');
  if (!this.session) {
    console.error('النموذج غير محمل');
    return;
  }

  try {
    const canvas = this.canvasElement.nativeElement;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('تعذر الحصول على context للكانفاس');
      return;
    }

    // تعيين أبعاد الكانفاس
    const video = this.videoElement.nativeElement;
    canvas.width = 320;
    canvas.height = 320;

    // رسم الفيديو على الكانفاس
    ctx.drawImage(video, 0, 0, 320, 320);

    // تحويل الصورة لتنسيق مناسب للنموذج
    const imageData = ctx.getImageData(0, 0, 320, 320);
    const inputTensor = this.preprocessImage(imageData);

    console.log('📥 اسم الإدخال المستخدم:', this.session.inputNames[0]);
    console.log('📤 اسم الإخراج المتوقع:', this.session.outputNames[0]);
    console.log('جاري تشغيل النموذج...');

    // ✅ تمرير الإدخال بالاسم الصحيح فعليًا
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.session.inputNames[0]] = inputTensor;

    // تشغيل النموذج
    const results = await this.session.run(feeds);

    console.log('✅ تم تشغيل النموذج بنجاح');
    console.log('نتائج النموذج:', results);

    // استخراج النتائج ومعالجتها
    const detections = this.postprocessResults(results);
    console.log('الكائنات المكتشفة:', detections);

    // عرض النتائج على الشاشة
    this.processDetections(detections);

  } catch (error) {
    console.error('خطأ في التعرف على الأشياء:', error);
  }
}


  // معالجة الصورة قبل إدخالها للنموذج (مصححة)
  private preprocessImage(imageData: ImageData): Tensor {
    const { data, width, height } = imageData;
    const input = new Float32Array(3 * 320 * 320);

    // YOLOv8 يتوقع صورة RGB مع تطبيع 0-1
    for (let i = 0; i < data.length; i += 4) {
      const pixelIndex = i / 4;
      const y = Math.floor(pixelIndex / width);
      const x = pixelIndex % width;

      if (x < 320 && y < 320) {
        const targetIndex = y * 320 + x;

        // قنوات RGB مع تطبيع
        input[targetIndex] = data[i] / 255.0; // R
        input[320 * 320 + targetIndex] = data[i + 1] / 255.0; // G
        input[2 * 320 * 320 + targetIndex] = data[i + 2] / 255.0; // B
      }
    }

    console.log('تمت معالجة الصورة، الأبعاد:', [1, 3, 320, 320]);
    return new Tensor('float32', input, [1, 3, 320, 320]);
  }

  // معالجة نتائج النموذج (مصححة)
  private postprocessResults(results: any): Detection[] {
    const detections: Detection[] = [];

    console.log('هيكل النتائج:', Object.keys(results));

    // البحث عن المخرج الصحيح - YOLOv8 له أسماء مخرجات مختلفة
    let outputTensor;
    const possibleOutputNames = ['output0', 'output', 'detections', 'boxes'];

    for (const name of possibleOutputNames) {
      if (results[name]) {
        outputTensor = results[name];
        console.log(`✅ تم العثور على المخرج باسم: ${name}`);
        break;
      }
    }

    if (!outputTensor) {
      // إذا لم نجد باسم معروف، نأخذ أول مخرج
      const firstKey = Object.keys(results)[0];
      outputTensor = results[firstKey];
      console.log(`🔶 استخدام المخرج الأول: ${firstKey}`);
    }

    const output = outputTensor.data;
    const outputDims = outputTensor.dims;

    console.log('أبعاد المخرج:', outputDims);
    console.log('بيانات المخرج:', output);

    // YOLOv8 عادة يعطي شكل [1, 84, 8400] أو [1, 5, 8400]
    // حيث 84 = 4 (bbox) + 80 (classes) أو 5 = 4 (bbox) + 1 (confidence)

    if (outputDims.length === 3 && outputDims[0] === 1) {
      const numClasses = outputDims[1] - 4; // نطرح 4 لإحداثيات bbox
      const numBoxes = outputDims[2];

      console.log(`عدد الصنوف: ${numClasses}, عدد المربعات: ${numBoxes}`);

      for (let i = 0; i < numBoxes; i++) {
        const startIdx = i * outputDims[1];
        const confidence = output[startIdx + 4];

        if (confidence > 0.5) {
          // العثور على الصنف ذو الثقة الأعلى
          let maxClassConfidence = 0;
          let maxClassId = 0;

          for (let j = 0; j < numClasses; j++) {
            const classConfidence = output[startIdx + 4 + j];
            if (classConfidence > maxClassConfidence) {
              maxClassConfidence = classConfidence;
              maxClassId = j;
            }
          }

          const finalConfidence = confidence * maxClassConfidence;

          if (finalConfidence > 0.5) {
            detections.push({
              class: this.classNames[maxClassId] || `كائن ${maxClassId}`,
              confidence: finalConfidence,
              bbox: [
                output[startIdx], // x
                output[startIdx + 1], // y
                output[startIdx + 2], // width
                output[startIdx + 3], // height
              ],
            });
          }
        }
      }
    }

    console.log(`تم اكتشاف ${detections.length} كائن`);
    return detections;
  }

  // معالجة الاكتشافات
  private processDetections(detections: Detection[]) {
    if (detections.length === 0) {
      this.lastDetectedObject = '';
      return;
    }

    // الحصول على أكثر كائن ثقة
    const bestDetection = detections.reduce((prev, current) =>
      prev.confidence > current.confidence ? prev : current
    );

    this.lastDetectedObject = `${bestDetection.class} (${Math.round(
      bestDetection.confidence * 100
    )}%)`;

    // نطق الكائن المكتشف
    this.speakObject(bestDetection.class);

    // اهتزاز الجهاز
    if (navigator.vibrate) {
      navigator.vibrate(200);
    }
  }

  // نطق الكائن المكتشف
  private speakObject(objectName: string) {
    const now = Date.now();

    // التحقق من عدم تكرار النطق لنفس الكائن في فترة قصيرة
    if (
      objectName === this.lastSpokenObject &&
      now - this.lastSpokenTime < this.SPEECH_COOLDOWN
    ) {
      return;
    }

    if ('speechSynthesis' in window) {
      const msg = new SpeechSynthesisUtterance(`${objectName} أمامك`);
      msg.lang = 'ar-SA';
      msg.rate = 1;
      msg.pitch = 1;

      window.speechSynthesis.speak(msg);

      this.lastSpokenObject = objectName;
      this.lastSpokenTime = now;
    }
  }

  // إعادة تحميل النموذج
  async reloadModel() {
    this.stopCamera();
    await this.loadModel();
  }

  // تبديل الوضع الليلي
  toggleDarkMode() {
    this.isDarkMode = !this.isDarkMode;
    document.documentElement.classList.toggle('dark', this.isDarkMode);
    localStorage.setItem('darkMode', this.isDarkMode.toString());
  }

  // تحميل تفضيلات الوضع
  private loadThemePreference() {
    const saved = localStorage.getItem('darkMode');
    this.isDarkMode = saved === 'true';
    document.documentElement.classList.toggle('dark', this.isDarkMode);
  }

  // اختبار النموذج بصورة ثابتة
  async testModelWithStaticImage() {
    if (!this.session) {
      console.error('النموذج غير محمل');
      return;
    }

    try {
      console.log('🔍 اختبار النموذج بصورة ثابتة...');

      const canvas = this.canvasElement.nativeElement;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // إنشاء صورة اختبارية (مربع أحمر)
      canvas.width = 320;
      canvas.height = 320;
      ctx.fillStyle = 'red';
      ctx.fillRect(100, 100, 50, 50);

      const imageData = ctx.getImageData(0, 0, 320, 320);
      const input = this.preprocessImage(imageData);

      // تشغيل النموذج
      const results = await this.session.run({
        [this.session.inputNames[0]]: input,
      });
      console.log('نتائج اختبار الصورة الثابتة:', results);

      const detections = this.postprocessResults(results);
      console.log('الاكتشافات من الصورة الثابتة:', detections);

      if (detections.length > 0) {
        this.processDetections(detections);
      } else {
        console.log('❌ لم يتم اكتشاف أي كائن في الصورة الاختبارية');
      }
    } catch (error) {
      console.error('خطأ في اختبار النموذج:', error);
    }
  }

  // تحميل صورة خارجية للاختبار
  async testModelWithExternalImage(imageUrl: string) {
    if (!this.session) {
      console.error('النموذج غير محمل');
      return;
    }

    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = async () => {
        try {
          const canvas = this.canvasElement.nativeElement;
          const ctx = canvas.getContext('2d');
          if (!ctx) return;

          canvas.width = 320;
          canvas.height = 320;
          ctx.drawImage(img, 0, 0, 320, 320);

          const imageData = ctx.getImageData(0, 0, 320, 320);
          const input = this.preprocessImage(imageData);

          const results = await this.session!.run({ images: input });
          console.log('نتائج اختبار الصورة الخارجية:', results);

          const detections = this.postprocessResults(results);
          console.log('الاكتشافات من الصورة الخارجية:', detections);

          this.processDetections(detections);
          resolve(detections);
        } catch (error) {
          console.error('خطأ في اختبار الصورة الخارجية:', error);
          resolve([]);
        }
      };
      img.src = imageUrl;
    });
  }
}
