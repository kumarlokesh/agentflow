package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/kumarlokesh/agentflow"
	"github.com/kumarlokesh/agentflow/agentflowtest"
	"github.com/kumarlokesh/agentflow/llm"
)

// DefaultInstructions is the calculator agent's system prompt.
const DefaultInstructions = "You are a careful calculator. Use the calculator tool for arithmetic and show the steps."

// NewCalculatorAgent builds the example agent. instructions lets the demo
// simulate a code change; model is the offline model below unless replaced.
func NewCalculatorAgent(instructions string, model llm.Model) (*agentflow.Agent, error) {
	if model == nil {
		model = offlineModel()
	}
	calc := agentflow.NewTool(llm.ToolSpec{
		Name:        "calculator",
		Description: "Evaluates an arithmetic expression with + - * / and parentheses.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
		Annotations: &llm.ToolAnnotations{ReadOnly: true, Idempotent: true},
	}, func(_ context.Context, in struct {
		Expression string `json:"expression"`
	}) (agentflow.ToolOutput, error) {
		v, err := evalExpression(in.Expression)
		if err != nil {
			return agentflow.ErrorOutput(err.Error()), nil
		}
		return agentflow.TextOutput(fmt.Sprint(v)), nil
	})
	return agentflow.NewAgent(agentflow.AgentConfig{
		Name:         "calculator",
		Model:        model,
		ModelName:    "offline-calculator",
		Instructions: instructions,
		Tools:        []agentflow.Tool{calc},
	})
}

// offlineModel stands in for a real model so the example runs without an
// API key: it asks for the calculator, then answers with the tool's result.
func offlineModel() llm.Model {
	return agentflowtest.ModelFunc(func(_ context.Context, req *llm.Request) (*llm.Response, error) {
		last := req.Messages[len(req.Messages)-1]
		for _, b := range last.Blocks {
			if r, ok := b.(llm.ToolResult); ok {
				answer := llm.TextOf(r.Content)
				return &llm.Response{
					Message:    llm.Message{Role: llm.RoleAssistant, Blocks: llm.Blocks{llm.Text{Text: "15 × 7 = 105, 23 − 8 = 15, 105 + 15 = " + answer + ". The answer is " + answer + "."}}},
					StopReason: llm.StopEndTurn,
					Usage:      llm.Usage{InputTokens: 80, OutputTokens: 40},
				}, nil
			}
		}
		expr := "(15*7)+(23-8)"
		if strings.Contains(llm.TextOf(last.Blocks), "2+2") {
			expr = "2+2"
		}
		return &llm.Response{
			Message:    llm.Message{Role: llm.RoleAssistant, Blocks: llm.Blocks{agentflowtest.Use("call_1", "calculator", map[string]string{"expression": expr})}},
			StopReason: llm.StopToolUse,
			Usage:      llm.Usage{InputTokens: 50, OutputTokens: 20},
		}, nil
	})
}

// evalExpression is a simple recursive-descent expression evaluator.
func evalExpression(expr string) (float64, error) {
	expr = strings.ReplaceAll(expr, " ", "")
	p := &parser{input: expr}
	result := p.parseExpr()
	if p.err != nil {
		return 0, p.err
	}
	if p.pos < len(p.input) {
		return 0, fmt.Errorf("unexpected character at position %d: %c", p.pos, p.input[p.pos])
	}
	return result, nil
}

type parser struct {
	input string
	pos   int
	err   error
}

func (p *parser) parseExpr() float64 {
	result := p.parseTerm()
	for p.pos < len(p.input) && (p.input[p.pos] == '+' || p.input[p.pos] == '-') {
		op := p.input[p.pos]
		p.pos++
		right := p.parseTerm()
		if op == '+' {
			result += right
		} else {
			result -= right
		}
	}
	return result
}

func (p *parser) parseTerm() float64 {
	result := p.parseFactor()
	for p.pos < len(p.input) && (p.input[p.pos] == '*' || p.input[p.pos] == '/') {
		op := p.input[p.pos]
		p.pos++
		right := p.parseFactor()
		if op == '*' {
			result *= right
		} else {
			if right == 0 {
				p.err = fmt.Errorf("division by zero")
				return 0
			}
			result /= right
		}
	}
	return result
}

func (p *parser) parseFactor() float64 {
	if p.pos >= len(p.input) {
		p.err = fmt.Errorf("unexpected end of expression")
		return 0
	}

	// Handle parentheses.
	if p.input[p.pos] == '(' {
		p.pos++ // skip '('
		result := p.parseExpr()
		if p.pos < len(p.input) && p.input[p.pos] == ')' {
			p.pos++ // skip ')'
		} else {
			p.err = fmt.Errorf("missing closing parenthesis")
		}
		return result
	}

	// Handle negative numbers.
	negative := false
	if p.input[p.pos] == '-' {
		negative = true
		p.pos++
	}

	// Parse number.
	start := p.pos
	for p.pos < len(p.input) && (p.input[p.pos] >= '0' && p.input[p.pos] <= '9' || p.input[p.pos] == '.') {
		p.pos++
	}
	if start == p.pos {
		p.err = fmt.Errorf("expected number at position %d", p.pos)
		return 0
	}

	num, err := strconv.ParseFloat(p.input[start:p.pos], 64)
	if err != nil {
		p.err = err
		return 0
	}
	if negative {
		num = -num
	}
	return num
}
