package com.siddu.offlinedocai

import android.content.ContextWrapper
import com.siddu.offlinedocai.ai.llm.GGUFLLM
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class PureAIChatTest {

    @Test
    fun testPureAIChatAnswersGeneralQuestionsWithoutRequiringDocument() = runBlocking {
        // GGUFLLM in fallback / general knowledge mode
        val dummyContext = ContextWrapper(null)
        val llm = GGUFLLM(dummyContext)

        // 1. Test greetings
        val greetingResp = llm.generate("Hello there!")
        assertTrue("Greeting should be responsive: $greetingResp", greetingResp.contains("Hello") || greetingResp.contains("How can I help"))
        assertFalse("Should never ask for document", greetingResp.contains("ensure a document is linked"))

        // 2. Test math calculation
        val mathResp = llm.generate("what is 25 * 4")
        assertTrue("Math response should contain result 100: $mathResp", mathResp.contains("100"))
        assertFalse("Should not mention document", mathResp.contains("ensure a document is linked"))

        val sqrtResp = llm.generate("sqrt(144)")
        assertTrue("Math response should contain 12: $sqrtResp", sqrtResp.contains("12"))

        val pctResp = llm.generate("20% of 150")
        assertTrue("Percentage should calculate 30: $pctResp", pctResp.contains("30"))

        // 3. Test programming request
        val codeResp = llm.generate("Write a python function to reverse a string")
        assertTrue("Code response should contain python snippet: $codeResp", codeResp.contains("def reverse_string") || codeResp.contains("[::-1]"))
        assertFalse("Should not mention document", codeResp.contains("ensure a document is linked"))

        val fibResp = llm.generate("fibonacci sequence in python")
        assertTrue("Fibonacci response should contain code: $fibResp", fibResp.contains("fibonacci"))

        val primeResp = llm.generate("how to check prime number")
        assertTrue("Prime check should contain code/algorithm: $primeResp", primeResp.contains("is_prime") || primeResp.contains("Prime"))

        // 4. Test science concept
        val scienceResp = llm.generate("Explain quantum computing")
        assertTrue("Science response should explain quantum concepts: $scienceResp", scienceResp.contains("Quantum") || scienceResp.contains("Qubit"))
        assertFalse("Should not mention document", scienceResp.contains("ensure a document is linked"))

        val photoResp = llm.generate("How does photosynthesis work?")
        assertTrue("Photosynthesis explanation should be detailed: $photoResp", photoResp.contains("Photosynthesis") || photoResp.contains("glucose") || photoResp.contains("light"))

        val relResp = llm.generate("Explain theory of relativity")
        assertTrue("Relativity explanation should mention Einstein or E=mc²: $relResp", relResp.contains("Relativity") || relResp.contains("Einstein") || relResp.contains("mc²"))

        // 5. Test writing / composition
        val emailResp = llm.generate("Draft an email follow up")
        assertTrue("Email draft should contain template: $emailResp", emailResp.contains("Subject:") || emailResp.contains("Dear"))

        val prosConsResp = llm.generate("Pros and cons of remote work")
        assertTrue("Pros and cons analysis should contain advantages and disadvantages: $prosConsResp", prosConsResp.contains("Advantages") || prosConsResp.contains("Pros"))

        // 6. Test arbitrary open-ended question
        val generalResp = llm.generate("What are the advantages of cloud computing?")
        assertTrue("General question response should provide structured insights: $generalResp", generalResp.isNotBlank())
        assertFalse("Should NEVER refuse with document requirement", generalResp.contains("ensure a document is linked"))
    }
}
