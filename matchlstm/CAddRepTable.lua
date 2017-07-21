
local CAddRepTable, parent = torch.class('nn.CAddRepTable', 'nn.Module')

function CAddRepTable:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CAddRepTable:updateOutput(input)
   self.output:resizeAs(input[1]):copy(input[1])
   for i=2,#input do
       for j = 1, self.output:size(1) do
           self.output[j]:add(input[i])
       end
   end
   return self.output
end

function CAddRepTable:updateGradInput(input, gradOutput)
   for i=1,#input do

      self.gradInput[i] = self.gradInput[i] or input[1].new()
      self.gradInput[i]:resizeAs(input[i])
      if i == 1 then
          self.gradInput[i]:copy(gradOutput)
      else
          self.gradInput[i]:copy(gradOutput:sum(1))
      end

   end

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end

   return self.gradInput
end
