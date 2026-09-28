package com.siddu.offlinedocai.ai.queue

data class QuestionQueueState(
    val isActive: Boolean = false,
    val isPaused: Boolean = false,
    val conversationId: Long? = null,
    val totalQuestions: Int = 0,
    val completedCount: Int = 0,
    val currentQuestion: String? = null,
    val pendingQuestions: List<String> = emptyList(),
    val completedQuestions: List<String> = emptyList(),
    val lastError: String? = null
) {
    val progress: Float
        get() = if (totalQuestions > 0) completedCount.toFloat() / totalQuestions else 0f

    val isCompleted: Boolean
        get() = totalQuestions > 0 && completedCount >= totalQuestions && !isActive
}
