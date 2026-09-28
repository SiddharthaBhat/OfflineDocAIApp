package com.siddu.offlinedocai.ai.queue

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

import kotlinx.coroutines.isActive

/**
 * Manages sequential execution of queued questions for a conversation.
 */
class QuestionQueueManager(
    private val scope: CoroutineScope
) {
    private val _queueState = MutableStateFlow(QuestionQueueState())
    val queueState: StateFlow<QuestionQueueState> = _queueState.asStateFlow()

    private var queueJob: Job? = null
    @Volatile private var isPausedFlag = false

    fun startQueue(
        conversationId: Long,
        questions: List<String>,
        docIds: List<Long>,
        onExecuteQuestion: suspend (convId: Long, question: String, docIds: List<Long>) -> Unit
    ) {
        if (questions.isEmpty()) return

        stopQueue()

        isPausedFlag = false
        _queueState.value = QuestionQueueState(
            isActive = true,
            isPaused = false,
            conversationId = conversationId,
            totalQuestions = questions.size,
            completedCount = 0,
            currentQuestion = questions.firstOrNull(),
            pendingQuestions = questions.drop(1),
            completedQuestions = emptyList()
        )

        queueJob = scope.launch(Dispatchers.IO) {
            val allQuestions = questions.toList()
            var compCount = 0

            for (i in allQuestions.indices) {
                if (!isActive) break

                // Pause check
                while (isPausedFlag && isActive) {
                    delay(250)
                }
                if (!isActive) break

                val question = allQuestions[i]
                val remaining = allQuestions.drop(i + 1)

                _queueState.value = _queueState.value.copy(
                    isActive = true,
                    isPaused = isPausedFlag,
                    conversationId = conversationId,
                    totalQuestions = allQuestions.size,
                    completedCount = compCount,
                    currentQuestion = question,
                    pendingQuestions = remaining
                )

                try {
                    // Sequentially process the question and await completion
                    onExecuteQuestion(conversationId, question, docIds)

                    compCount++
                    val completed = _queueState.value.completedQuestions + question

                    _queueState.value = _queueState.value.copy(
                        completedCount = compCount,
                        completedQuestions = completed
                    )

                    // Small pause between questions for smooth DB sync and UI update
                    if (i < allQuestions.size - 1) {
                        delay(500)
                    }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    _queueState.value = _queueState.value.copy(
                        lastError = e.message ?: "Failed on: $question"
                    )
                }
            }

            _queueState.value = _queueState.value.copy(
                isActive = false,
                isPaused = false,
                currentQuestion = null,
                pendingQuestions = emptyList()
            )
        }
    }

    fun pauseQueue() {
        if (_queueState.value.isActive) {
            isPausedFlag = true
            _queueState.value = _queueState.value.copy(isPaused = true)
        }
    }

    fun resumeQueue() {
        if (_queueState.value.isActive && _queueState.value.isPaused) {
            isPausedFlag = false
            _queueState.value = _queueState.value.copy(isPaused = false)
        }
    }

    fun stopQueue() {
        queueJob?.cancel()
        queueJob = null
        isPausedFlag = false
        _queueState.value = _queueState.value.copy(
            isActive = false,
            isPaused = false,
            currentQuestion = null,
            pendingQuestions = emptyList()
        )
    }

    fun clearQueue() {
        stopQueue()
        _queueState.value = QuestionQueueState()
    }
}
