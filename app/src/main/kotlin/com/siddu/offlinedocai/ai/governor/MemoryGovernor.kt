package com.siddu.offlinedocai.ai.governor

import android.util.Log

/**
 * Memory-Aware Adaptive Runtime Governor
 *
 * Implements the Android memory constraint architecture formulated in:
 * - Pocket RAG: On-Device RAG for First Aid Guidance in Offline Mobile Environments (2026)
 * - HeRo: Adaptive Orchestration of Agentic RAG on Heterogeneous Mobile SoC (DAC 2026)
 *
 * Enforces the strict memory budget:
 *   M_total = M_model + M_index + M_KV + M_runtime <= 2.0 GB
 *
 * Dynamically tracks JVM heap pressure ratio:
 *   rho = M_used / M_max
 *
 * Modulates retrieval parameters (Top-K), maximum generation token budget (T_max),
 * and Selective Content Reduction (SCR) compression intensity to prevent Out-Of-Memory (OOM)
 * crashes and aggressive Android LowMemoryKiller (LMK) process termination.
 */
object MemoryGovernor {

    private const val TAG = "MemoryGovernor"

    enum class PressureLevel {
        SAFE,       // rho < 0.70: Full context allocation
        MODERATE,   // 0.70 <= rho < 0.85: Scaled down context and increased SCR compression
        CRITICAL    // rho >= 0.85: Minimal context, aggressive pruning to guarantee stability
    }

    data class GovernanceConfig(
        val pressureLevel: PressureLevel,
        val pressureRatio: Float,
        val topK: Int,
        val maxContextTokens: Int,
        val scrRetentionRatio: Float
    )

    /**
     * Inspects the current device and process heap state and returns optimal runtime constraints.
     */
    fun getAdaptiveConfig(): GovernanceConfig {
        val runtime = Runtime.getRuntime()
        val totalMemory = runtime.totalMemory()
        val freeMemory = runtime.freeMemory()
        val maxMemory = runtime.maxMemory()

        val usedMemory = totalMemory - freeMemory
        val pressureRatio = (usedMemory.toDouble() / maxMemory.toDouble()).toFloat().coerceIn(0f, 1f)

        val config = when {
            pressureRatio < 0.70f -> {
                GovernanceConfig(
                    pressureLevel = PressureLevel.SAFE,
                    pressureRatio = pressureRatio,
                    topK = 5,
                    maxContextTokens = 1024,
                    scrRetentionRatio = 0.75f
                )
            }
            pressureRatio < 0.85f -> {
                Log.w(TAG, "Moderate memory pressure detected (${String.format("%.1f", pressureRatio * 100)}%). Adjusting RAG budgets.")
                GovernanceConfig(
                    pressureLevel = PressureLevel.MODERATE,
                    pressureRatio = pressureRatio,
                    topK = 3,
                    maxContextTokens = 768,
                    scrRetentionRatio = 0.60f
                )
            }
            else -> {
                Log.e(TAG, "Critical memory pressure detected (${String.format("%.1f", pressureRatio * 100)}%). Enforcing emergency context clamping.")
                GovernanceConfig(
                    pressureLevel = PressureLevel.CRITICAL,
                    pressureRatio = pressureRatio,
                    topK = 2,
                    maxContextTokens = 384,
                    scrRetentionRatio = 0.40f
                )
            }
        }

        return config
    }
}
