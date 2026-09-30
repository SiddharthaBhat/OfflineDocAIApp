package org.nehuatl.llamacpp;

import android.os.Build;
import android.os.ParcelFileDescriptor;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Native JNI bridge for llama.cpp execution on Android.
 * Communicates with librnllama.so across ARM64-v8a (with dotprod, i8mm acceleration) and x86_64.
 */
public class LlamaContext {

    private static final String TAG = "RNLlamaContext";
    private static boolean isLibraryLoaded = false;

    static {
        loadNativeLibraries();
    }

    private static void loadNativeLibraries() {
        if (isLibraryLoaded) return;
        try {
            String[] supportedAbis = Build.SUPPORTED_ABIS;
            String primaryAbi = (supportedAbis != null && supportedAbis.length > 0) ? supportedAbis[0] : "";

            if (primaryAbi.contains("arm64")) {
                String cpuInfo = getCpuInfo();
                boolean isUnsafeDotprod = isUnsafeDotprodDevice(cpuInfo);
                boolean hasDotprod = !isUnsafeDotprod && (cpuInfo.contains("dotprod") || cpuInfo.contains("asimddp"));
                boolean hasI8mm = !isUnsafeDotprod && cpuInfo.contains("i8mm");
                boolean hasV82 = !isUnsafeDotprod && cpuInfo.contains("asimd") && cpuInfo.contains("crc32") && cpuInfo.contains("aes");

                if (isUnsafeDotprod) {
                    Log.i(TAG, "MediaTek Helio / non-standard dotprod chipset detected. Using stable baseline librnllama_v8.so");
                    try {
                        System.loadLibrary("rnllama_v8");
                    } catch (UnsatisfiedLinkError e) {
                        Log.i(TAG, "Fallback loading librnllama.so");
                        System.loadLibrary("rnllama");
                    }
                } else if (hasV82 && hasDotprod && hasI8mm) {
                    Log.i(TAG, "Loading librnllama_v8_2_dotprod_i8mm.so");
                    System.loadLibrary("rnllama_v8_2_dotprod_i8mm");
                } else if (hasV82 && hasDotprod) {
                    Log.i(TAG, "Loading librnllama_v8_2_dotprod.so");
                    System.loadLibrary("rnllama_v8_2_dotprod");
                } else if (hasV82 && hasI8mm) {
                    Log.i(TAG, "Loading librnllama_v8_2_i8mm.so");
                    System.loadLibrary("rnllama_v8_2_i8mm");
                } else if (hasV82) {
                    Log.i(TAG, "Loading librnllama_v8_2.so");
                    System.loadLibrary("rnllama_v8_2");
                } else {
                    try {
                        Log.i(TAG, "Loading librnllama_v8.so");
                        System.loadLibrary("rnllama_v8");
                    } catch (UnsatisfiedLinkError e) {
                        Log.i(TAG, "Fallback loading librnllama.so");
                        System.loadLibrary("rnllama");
                    }
                }
            } else if (primaryAbi.contains("x86_64")) {
                try {
                    Log.i(TAG, "Loading librnllama_x86_64.so");
                    System.loadLibrary("rnllama_x86_64");
                } catch (UnsatisfiedLinkError e) {
                    Log.i(TAG, "Fallback loading librnllama.so");
                    System.loadLibrary("rnllama");
                }
            } else {
                Log.i(TAG, "Loading default librnllama.so");
                System.loadLibrary("rnllama");
            }
            isLibraryLoaded = true;
        } catch (Throwable t) {
            Log.e(TAG, "Failed to load specialized native llama library, falling back to default rnllama", t);
            try {
                System.loadLibrary("rnllama");
                isLibraryLoaded = true;
            } catch (Throwable t2) {
                Log.e(TAG, "Fatal: unable to load librnllama.so", t2);
            }
        }
    }

    private static boolean isUnsafeDotprodDevice(String cpuInfo) {
        String hardware = (Build.HARDWARE != null ? Build.HARDWARE : "").toLowerCase();
        String board = (Build.BOARD != null ? Build.BOARD : "").toLowerCase();
        String cpu = cpuInfo.toLowerCase();

        // MediaTek MT67xx Helio G/P series (e.g. MT6768, MT6769, MT6785, MT6771, MT6779, Helio G80/G85/G88/G90/G95/P60/P65)
        return hardware.contains("mt676") || hardware.contains("mt677") ||
                hardware.contains("mt678") || board.contains("mt676") || board.contains("mt677") ||
                board.contains("mt678") || cpu.contains("mt676") || cpu.contains("mt677") ||
                cpu.contains("mt678") || cpu.contains("helio");
    }

    private static String getCpuInfo() {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader("/proc/cpuinfo"))) {
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line.toLowerCase()).append(" ");
            }
        } catch (Exception ignored) {
        }
        return sb.toString();
    }

    public static boolean isSupported() {
        return isLibraryLoaded;
    }

    // Instance fields
    private final int id;
    private long contextPointer = 0L;
    private boolean isReleased = false;

    public interface TokenCallback {
        void onToken(String token);
    }

    private TokenCallback tokenCallback = null;

    /**
     * Inner callback matching the exact JNI signature expected by librnllama.so
     */
    public class PartialCompletionCallback {
        private final boolean emitNeeded;

        public PartialCompletionCallback(boolean emitNeeded) {
            this.emitNeeded = emitNeeded;
        }

        public void onPartialCompletion(Map<String, Object> tokenDict) {
            if (emitNeeded && tokenDict != null && tokenCallback != null) {
                Object tokenObj = tokenDict.get("token");
                if (tokenObj != null) {
                    tokenCallback.onToken(tokenObj.toString());
                }
            }
        }
    }

    public LlamaContext(int id, File modelFile, int nCtx, int nBatch, int nThreads, boolean embedding) throws Exception {
        this.id = id;
        if (!isLibraryLoaded) {
            throw new IllegalStateException("Native llama library is not loaded on this architecture.");
        }
        if (!modelFile.exists()) {
            throw new IllegalArgumentException("Model file does not exist: " + modelFile.getAbsolutePath());
        }

        ParcelFileDescriptor pfd = ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY);
        // CRITICAL FOR ANDROID FDSAN:
        // librnllama calls close(fd) in native C++ (Java_org_nehuatl_llamacpp_LlamaContext_initContextWithFd).
        // If the fd is owned by a Java ParcelFileDescriptor, Android's libc fdsan detects an untagged
        // native close() and immediately aborts the process with SIGABRT:
        // "fdsan: attempted to close file descriptor, expected to be unowned, actually owned by ParcelFileDescriptor".
        // Calling pfd.detachFd() transfers ownership and clears the fdsan tag so native close(fd) succeeds cleanly.
        int fd = pfd.detachFd();

        int[] emptyIntArray = new int[0];
        // JNI call: initContextWithFd
        // Signature: (int fd, boolean isGGUF, int nCtx, int nBatch, int nThreads, int nGpuLayers,
        //             boolean useMmap, boolean useMlock, boolean embedding, String lora,
        //             float loraScaled, float ropeFreqBase, float ropeFreqScale, int mmprojFd, int[] imageFds)
        // CRUCIAL: mmprojFd must be -1 when no multimodal vision projection model is provided.
        this.contextPointer = initContextWithFd(
                fd,
                true,               // isGGUF
                nCtx,               // nCtx
                nBatch,             // nBatch
                nThreads,           // nThreads
                0,                  // nGpuLayers (0 for mobile CPU stability)
                true,               // useMmap
                false,              // useMlock
                embedding,          // embedding
                "",                 // lora
                1.0f,               // loraScaled
                0.0f,               // rope_freq_base (0.0f = default model value)
                0.0f,               // rope_freq_scale (0.0f = default model value)
                -1,                 // mmproj_fd (-1 = no multimodal model)
                emptyIntArray       // image_fds (empty int[0])
        );

        if (this.contextPointer == 0L) {
            throw new RuntimeException("Failed to initialize llama.cpp native context from file: " + modelFile.getName());
        }
    }

    public synchronized String generateCompletion(
            String prompt,
            float temperature,
            int topK,
            float topP,
            int maxTokens,
            List<String> stopWords,
            TokenCallback tokenCallback
    ) {
        if (isReleased || contextPointer == 0L) {
            throw new IllegalStateException("Context is released or invalid");
        }

        final StringBuilder tokenAccumulator = new StringBuilder();
        this.tokenCallback = new TokenCallback() {
            @Override
            public void onToken(String token) {
                tokenAccumulator.append(token);
                if (tokenCallback != null) {
                    tokenCallback.onToken(token);
                }
            }
        };
        String[] stopArray = (stopWords != null) ? stopWords.toArray(new String[0]) : new String[0];
        // ALWAYS pass true so JNI invokes onPartialCompletion and captures all tokens
        PartialCompletionCallback jniCallback = new PartialCompletionCallback(true);

        try {
            Map<String, Object> result = doCompletion(
                    contextPointer,
                    prompt,
                    "",                 // grammar
                    temperature,        // temperature
                    4,                  // nThreads
                    maxTokens,          // nPredict
                    0,                  // nProbs
                    64,                 // penaltyLastN
                    1.1f,               // penaltyRepeat
                    0.0f,               // penaltyFreq
                    0.0f,               // penaltyPresent
                    0.0f,               // mirostat
                    5.0f,               // mirostatTau
                    0.1f,               // mirostatEta
                    false,              // penalizeNl
                    topK,               // topK
                    topP,               // topP
                    0.05f,              // minP
                    1.0f,               // tfsZ
                    1.0f,               // typicalP
                    0.0f,               // temperatureInc
                    1.0f,               // dynatempExponent
                    0,                  // dynatempRange
                    stopArray,          // stop
                    false,              // ignoreEos
                    new double[0][0],   // logitBias
                    new int[0],         // tokenBias
                    jniCallback
            );

            Log.i(TAG, "Native doCompletion finished. Keys: " + (result != null ? result.keySet() : "null")
                    + ", tokenAccumulator length: " + tokenAccumulator.length());

            if (result != null && result.containsKey("text")) {
                String text = String.valueOf(result.get("text"));
                if (text != null && !text.trim().isEmpty()) {
                    return text;
                }
            }
            if (tokenAccumulator.length() > 0) {
                return tokenAccumulator.toString();
            }
        } catch (Throwable t) {
            Log.e(TAG, "Native doCompletion failed", t);
            if (tokenAccumulator.length() > 0) {
                Log.i(TAG, "Recovered partial completion from tokenAccumulator despite error: " + tokenAccumulator.length() + " chars");
                return tokenAccumulator.toString();
            }
            throw new RuntimeException("Native LLM generation error: " + t.getMessage(), t);
        } finally {
            this.tokenCallback = null;
        }
        return "";
    }

    public synchronized float[] getEmbeddingVector(String text) {
        if (isReleased || contextPointer == 0L) {
            throw new IllegalStateException("Context is released or invalid");
        }
        try {
            if (!isEmbeddingEnabled(contextPointer)) {
                Log.w(TAG, "Embedding is not enabled for context pointer: " + contextPointer);
                return new float[0];
            }
            Map<String, Object> result = embedding(contextPointer, text);
            if (result != null && result.containsKey("embedding")) {
                Object embObj = result.get("embedding");
                if (embObj instanceof double[]) {
                    double[] dArr = (double[]) embObj;
                    float[] fArr = new float[dArr.length];
                    for (int i = 0; i < dArr.length; i++) {
                        fArr[i] = (float) dArr[i];
                    }
                    return fArr;
                } else if (embObj instanceof float[]) {
                    return (float[]) embObj;
                } else if (embObj instanceof List<?>) {
                    List<?> list = (List<?>) embObj;
                    float[] fArr = new float[list.size()];
                    for (int i = 0; i < list.size(); i++) {
                        Object item = list.get(i);
                        if (item instanceof Number) {
                            fArr[i] = ((Number) item).floatValue();
                        }
                    }
                    return fArr;
                }
            }
        } catch (Throwable t) {
            Log.w(TAG, "Native embedding calculation error", t);
        }
        return new float[0];
    }

    public synchronized void stop() {
        if (!isReleased && contextPointer != 0L) {
            stopCompletion(contextPointer);
        }
    }

    public synchronized void release() {
        if (!isReleased) {
            isReleased = true;
            if (contextPointer != 0L) {
                try {
                    freeContext(contextPointer);
                } catch (Throwable t) {
                    Log.w(TAG, "Error freeing native llama context", t);
                }
                contextPointer = 0L;
            }
        }
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            release();
        } finally {
            super.finalize();
        }
    }

    // --- Native JNI declarations matching librnllama.so ---
    private native long initContextWithFd(int fd, boolean isGGUF, int nCtx, int nBatch, int nThreads, int nGpuLayers,
                                          boolean useMmap, boolean useMlock, boolean embedding, String lora,
                                          float loraScaled, float ropeFreqBase, float ropeFreqScale, int mmprojFd, int[] imageFds);

    private native Map<String, Object> loadModelDetails(long context);

    public native String getFormattedChat(long context, Map<String, Object>[] messages, String chatTemplate);

    private native Map<String, Object> loadSession(long context, String path);

    private native int saveSession(long context, String path, int size);

    private native Map<String, Object> doCompletion(long context, String prompt, String grammar, float temperature,
                                                    int nThreads, int nPredict, int nProbs, int penaltyLastN,
                                                    float penaltyRepeat, float penaltyFreq, float penaltyPresent,
                                                    float mirostat, float mirostatTau, float mirostatEta,
                                                    boolean penalizeNl, int topK, float topP, float minP, float tfsZ,
                                                    float typicalP, float temperatureInc, float dynatempExponent,
                                                    int dynatempRange, String[] stop, boolean ignoreEos,
                                                    double[][] logitBias, int[] tokenBias,
                                                    PartialCompletionCallback callback);

    private native void stopCompletion(long context);

    private native boolean isPredicting(long context);

    private native List<Object> tokenize(long context, String text);

    private native String detokenize(long context, int[] tokens);

    private native boolean isEmbeddingEnabled(long context);

    private native Map<String, Object> embedding(long context, String text);

    private native String bench(long context, int pp, int tg, int pl, int nr);

    private native void freeContext(long context);
}
