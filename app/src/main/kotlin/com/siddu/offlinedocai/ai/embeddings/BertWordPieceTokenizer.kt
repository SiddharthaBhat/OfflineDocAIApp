package com.siddu.offlinedocai.ai.embeddings

import android.content.Context
import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader

data class TokenizedResult(
    val inputIds: LongArray,
    val attentionMask: LongArray,
    val tokenTypeIds: LongArray
)

class BertWordPieceTokenizer(context: Context) {
    private val TAG = "BertTokenizer"
    private val vocab = HashMap<String, Long>(32000)
    private val unkToken = "[UNK]"
    private val unkId: Long
    private val clsId: Long
    private val sepId: Long

    init {
        var loaded = 0
        try {
            context.assets.open("bert_vocab.txt").use { input ->
                BufferedReader(InputStreamReader(input, Charsets.UTF_8)).useLines { lines ->
                    var idx = 0L
                    lines.forEach { line ->
                        val token = line.trim()
                        if (token.isNotEmpty()) {
                            vocab[token] = idx
                            loaded++
                        }
                        idx++
                    }
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to load bert_vocab.txt from assets", t)
        }

        unkId = vocab[unkToken] ?: 100L
        clsId = vocab["[CLS]"] ?: 101L
        sepId = vocab["[SEP]"] ?: 102L
        Log.i(TAG, "BertWordPieceTokenizer initialized with $loaded vocabulary tokens")
    }

    fun tokenize(text: String, maxLen: Int = 512): TokenizedResult {
        val tokens = mutableListOf<Long>()
        tokens.add(clsId)

        val words = basicTokenize(text)
        for (word in words) {
            if (tokens.size >= maxLen - 1) break
            wordPieceTokenize(word, tokens, maxLen - 1)
        }

        tokens.add(sepId)

        val size = tokens.size
        val inputIds = LongArray(size) { tokens[it] }
        val attentionMask = LongArray(size) { 1L }
        val tokenTypeIds = LongArray(size) { 0L }

        return TokenizedResult(inputIds, attentionMask, tokenTypeIds)
    }

    private fun basicTokenize(text: String): List<String> {
        val result = mutableListOf<String>()
        val cleaned = text.lowercase()
        val current = StringBuilder()

        for (ch in cleaned) {
            if (ch.isWhitespace()) {
                if (current.isNotEmpty()) {
                    result.add(current.toString())
                    current.clear()
                }
            } else if (isPunctuation(ch)) {
                if (current.isNotEmpty()) {
                    result.add(current.toString())
                    current.clear()
                }
                result.add(ch.toString())
            } else {
                current.append(ch)
            }
        }
        if (current.isNotEmpty()) {
            result.add(current.toString())
        }
        return result
    }

    private fun wordPieceTokenize(word: String, outTokens: MutableList<Long>, maxCapacity: Int) {
        if (word.length > 100) {
            outTokens.add(unkId)
            return
        }

        var isBad = false
        var start = 0
        val subTokens = mutableListOf<Long>()

        while (start < word.length) {
            var end = word.length
            var curSubstr: String? = null
            var curId: Long? = null

            while (start < end) {
                var substr = word.substring(start, end)
                if (start > 0) {
                    substr = "##$substr"
                }
                if (vocab.containsKey(substr)) {
                    curSubstr = substr
                    curId = vocab[substr]
                    break
                }
                end--
            }

            if (curSubstr == null || curId == null) {
                isBad = true
                break
            }

            subTokens.add(curId)
            start = end
        }

        if (isBad) {
            outTokens.add(unkId)
        } else {
            for (tok in subTokens) {
                if (outTokens.size >= maxCapacity) break
                outTokens.add(tok)
            }
        }
    }

    private fun isPunctuation(ch: Char): Boolean {
        val type = Character.getType(ch)
        return type == Character.CONNECTOR_PUNCTUATION.toInt() ||
                type == Character.DASH_PUNCTUATION.toInt() ||
                type == Character.START_PUNCTUATION.toInt() ||
                type == Character.END_PUNCTUATION.toInt() ||
                type == Character.INITIAL_QUOTE_PUNCTUATION.toInt() ||
                type == Character.FINAL_QUOTE_PUNCTUATION.toInt() ||
                type == Character.OTHER_PUNCTUATION.toInt() ||
                (ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    }
}
