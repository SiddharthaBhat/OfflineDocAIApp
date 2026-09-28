package com.siddu.offlinedocai

import com.siddu.offlinedocai.ai.queue.QuestionQueueManager
import com.siddu.offlinedocai.ai.queue.QuestionQueueParser
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class QuestionQueueTest {

    @Test
    fun testParseJsonArrayOfStrings() {
        val json = """
            [
              "What is grafting?",
              "What tools are needed?",
              "How to perform cleft grafting?"
            ]
        """.trimIndent()

        val parsed = QuestionQueueParser.parseQuestions(json)
        assertEquals(3, parsed.size)
        assertEquals("What is grafting?", parsed[0])
        assertEquals("What tools are needed?", parsed[1])
        assertEquals("How to perform cleft grafting?", parsed[2])
    }

    @Test
    fun testParseJsonArrayOfObjects() {
        val json = """
            [
              {"question": "Explain rootstock selection."},
              {"q": "What is the cambium layer?"},
              {"prompt": "How to care for a graft union?"},
              {"text": "What causes graft incompatibility?"}
            ]
        """.trimIndent()

        val parsed = QuestionQueueParser.parseQuestions(json)
        assertEquals(4, parsed.size)
        assertEquals("Explain rootstock selection.", parsed[0])
        assertEquals("What is the cambium layer?", parsed[1])
        assertEquals("How to care for a graft union?", parsed[2])
        assertEquals("What causes graft incompatibility?", parsed[3])
    }

    @Test
    fun testParseJsonObjectWithQuestionsArray() {
        val json = """
            {
              "questions": [
                "Summarize chapter 1",
                "Explain page 5",
                "List common troubleshooting steps"
              ]
            }
        """.trimIndent()

        val parsed = QuestionQueueParser.parseQuestions(json)
        assertEquals(3, parsed.size)
        assertEquals("Summarize chapter 1", parsed[0])
        assertEquals("Explain page 5", parsed[1])
        assertEquals("List common troubleshooting steps", parsed[2])
    }

    @Test
    fun testParseNewlineDelimitedFallback() {
        val text = """
            1. First question about grafting?
            2. Second question about cambium?
            - Third question about timing?
            * Fourth question about tools?
        """.trimIndent()

        val parsed = QuestionQueueParser.parseQuestions(text)
        assertEquals(4, parsed.size)
        assertEquals("First question about grafting?", parsed[0])
        assertEquals("Second question about cambium?", parsed[1])
        assertEquals("Third question about timing?", parsed[2])
        assertEquals("Fourth question about tools?", parsed[3])
    }

    @Test
    fun testParseEmptyAndMalformedInput() {
        assertTrue(QuestionQueueParser.parseQuestions("").isEmpty())
        assertTrue(QuestionQueueParser.parseQuestions("   ").isEmpty())
        assertTrue(QuestionQueueParser.parseQuestions("[\n]\n").isEmpty())
        assertTrue(QuestionQueueParser.parseQuestions("{\n}\n").isEmpty())
    }

    @Test
    fun testCleanQuestionFormatting() {
        assertEquals("Clean question", QuestionQueueParser.cleanQuestion("\"Clean question\""))
        assertEquals("Clean question", QuestionQueueParser.cleanQuestion("'Clean question',"))
        assertEquals("Clean question", QuestionQueueParser.cleanQuestion("1. Clean question"))
        assertEquals("Clean question", QuestionQueueParser.cleanQuestion("1) Clean question"))
        assertEquals("Clean question", QuestionQueueParser.cleanQuestion("- Clean question"))
        assertEquals("Clean question", QuestionQueueParser.cleanQuestion("* Clean question"))
    }

    @Test
    fun testSequentialExecutionOrder() = runBlocking {
        val scope = CoroutineScope(Dispatchers.Default)
        val manager = QuestionQueueManager(scope)

        val questions = listOf("Question A", "Question B", "Question C")
        val executedList = mutableListOf<String>()
        val latch = CountDownLatch(3)

        manager.startQueue(
            conversationId = 1L,
            questions = questions,
            docIds = listOf(10L)
        ) { convId, q, docs ->
            assertEquals(1L, convId)
            assertEquals(listOf(10L), docs)
            synchronized(executedList) {
                executedList.add(q)
            }
            latch.countDown()
        }

        assertTrue("Execution timed out", latch.await(6, TimeUnit.SECONDS))
        delay(300)

        // Verify sequential completion of all questions
        synchronized(executedList) {
            assertEquals(3, executedList.size)
            assertEquals("Question A", executedList[0])
            assertEquals("Question B", executedList[1])
            assertEquals("Question C", executedList[2])
        }

        val finalState = manager.queueState.value
        assertFalse(finalState.isActive)
        assertEquals(3, finalState.totalQuestions)
        assertEquals(3, finalState.completedCount)
        assertTrue(finalState.isCompleted)
    }

    @Test
    fun testStopQueueCancellation() = runBlocking {
        val scope = CoroutineScope(Dispatchers.Default)
        val manager = QuestionQueueManager(scope)

        val questions = listOf("Question 1", "Question 2", "Question 3")
        val executedList = mutableListOf<String>()
        val latch = CountDownLatch(1)

        manager.startQueue(
            conversationId = 1L,
            questions = questions,
            docIds = emptyList()
        ) { _, q, _ ->
            synchronized(executedList) {
                executedList.add(q)
            }
            if (q == "Question 1") {
                manager.stopQueue()
                latch.countDown()
            }
        }

        assertTrue(latch.await(3, TimeUnit.SECONDS))
        delay(600)

        // Only first question was processed before cancellation
        synchronized(executedList) {
            assertEquals(1, executedList.size)
            assertEquals("Question 1", executedList[0])
        }
        assertFalse(manager.queueState.value.isActive)
    }
}
