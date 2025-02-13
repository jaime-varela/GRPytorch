import torch
import torch.nn as nn

# TODO: fix the evaluation of tensors.  The ricci tensor tensor for schwarchild should be zero.
# the reason for the error is likely due to numerical error from second order auto-differentiation.
# Also, references are needed to the tensor conventions in use


class SpaceTimeMetricModule(nn.Module):
    def __init__(self,metric_coordinate_function: callable):
        super(SpaceTimeMetricModule, self).__init__()
        self.metric_coordinate_function = metric_coordinate_function

    def forward(self, input_batch:torch.Tensor):
        # input_batch has shape (N, D), where N is the batch size and D is the dimensionality of the vectors
        N, D = input_batch.size()
        
        # Apply the vector_to_matrix_func to each vector in the batch
        matrices = torch.stack([self.metric_coordinate_function(vector) for vector in input_batch])
        
        # Reshape the matrices to have size (N, R, C)
        matrices = matrices.view(N, D, D)
        
        return matrices
    
class ChristoffelSymbols(nn.Module):
    '''
        Computes the christoffel symbols given by $\Gamma^{i}_{kl}$
    '''
    def __init__(self,metric_module: SpaceTimeMetricModule):
        super(ChristoffelSymbols, self).__init__()
        self.metric = metric_module

    # we also provide a function for single coordinate evaluations
    def christoffel_symbols(self,coordinate_vector:torch.Tensor):
        # look into jacfw instead of this jacobian
        met_jacobian = torch.autograd.functional.jacobian(self.metric.metric_coordinate_function,
                                                          coordinate_vector,
                                                          create_graph=True)

        return self._christoffel_symbols(coordinate_vector,met_jacobian)

    def _christoffel_symbols(self,coordinate_vector:torch.Tensor,met_jacobian:torch.Tensor):
        metric_schwar = self.metric.metric_coordinate_function(coordinate_vector)
        inverse_metric = torch.linalg.inv(metric_schwar)
        half_inv_met = 0.5*inverse_metric

        term_1 = torch.einsum('im,mkl->ikl',half_inv_met,met_jacobian)
        term_2 = torch.einsum('im,mlk->ikl',half_inv_met,met_jacobian)
        term_3 = torch.einsum('im,klm->ikl',half_inv_met,met_jacobian)
        return term_1 + term_2 - term_3


    def forward(self,coordinate_input: torch.Tensor):
        # input_batch has shape (N, D), where N is the batch size and D is the dimensionality of the vectors
        N, D = coordinate_input.size()

        metric_eval = self.metric(coordinate_input)
        inverse_metric = torch.linalg.inv(metric_eval)
        half_inv_met = 0.5*inverse_metric

        # Compute the derivative of the dependency module output
        # with respect to the input x using autograd
        met_jacobian = torch.func.vmap(torch.func.jacrev(self.metric.metric_coordinate_function))(coordinate_input)

        # n is the batch coordinate dimention
        term_1 = torch.einsum('nim,nmkl->nikl',half_inv_met,met_jacobian)
        term_2 = torch.einsum('nim,nmlk->nikl',half_inv_met,met_jacobian)
        term_3 = torch.einsum('nim,nklm->nikl',half_inv_met,met_jacobian)
        return term_1 + term_2 - term_3


# A utility to compute the derivative of the Christoffel symbols and other objects
# the goal is to significantly reduce jacobian coputations
class _ChristoffelDerivatives(nn.Module):
    def __init__(self,metric_module: SpaceTimeMetricModule):
        super(_ChristoffelDerivatives, self).__init__()
        self.metric = metric_module

    def forward(self,coordinate_input: torch.Tensor):
        N, D = coordinate_input.size()

        metric_eval = self.metric(coordinate_input)
        inverse_metric = torch.linalg.inv(metric_eval)
        half_inv_met = 0.5*inverse_metric

        # Compute the derivative of the dependency module output
        # with respect to the input x using autograd
        met_jacobian = torch.func.vmap(torch.func.jacrev(self.metric.metric_coordinate_function))(coordinate_input)

        # n is the batch coordinate dimention
        ch_term_1 = torch.einsum('nim,nmkl->nikl',half_inv_met,met_jacobian)
        ch_term_2 = torch.einsum('nim,nmlk->nikl',half_inv_met,met_jacobian)
        ch_term_3 = torch.einsum('nim,nklm->nikl',half_inv_met,met_jacobian)
        christoffel_symbol = ch_term_1 + ch_term_2 - ch_term_3

        metric_hessian = torch.func.vmap(torch.func.jacrev(torch.func.jacrev(self.metric.metric_coordinate_function)))(coordinate_input)

        hlaf_neg_met = -0.5 * inverse_metric
        deriv_inv_met = torch.einsum('nas,nsmd,nmb->nabd',hlaf_neg_met,met_jacobian,inverse_metric)
        chdf_term_1 = torch.einsum('nimd,nmkl->nikld',deriv_inv_met,met_jacobian)
        chdf_term_2 = torch.einsum('nimd,nmlk->nikld',deriv_inv_met,met_jacobian)
        chdf_term_3 = torch.einsum('nimd,nklm->nikld',deriv_inv_met,met_jacobian)

        chd_term_1 = torch.einsum('nim,nmkld->nikld',half_inv_met,metric_hessian)
        chd_term_2 = torch.einsum('nim,nmlkd->nikld',half_inv_met,metric_hessian)
        chd_term_3 = torch.einsum('nim,nklmd->nikld',half_inv_met,metric_hessian)
        christoffel_derivative = (chdf_term_1 + chdf_term_2 - chdf_term_3) + (chd_term_1 + chd_term_2 - chd_term_3)

        inverse_metric_derivative = -2.0 * deriv_inv_met
        return christoffel_symbol, met_jacobian, christoffel_derivative, metric_hessian, inverse_metric_derivative

class RiemannTensor(nn.Module):
    '''
        Computes the Riemann tensor given by $\R^{p}_{smv}$
    '''
    def __init__(self,metric_module: SpaceTimeMetricModule):
        super(RiemannTensor, self).__init__()
        self.metric = metric_module
        self._christoffel_derv = _ChristoffelDerivatives(metric_module)

    def forward(self,coordinate_input: torch.Tensor):
        # input_batch has shape (N, D), where N is the batch size and D is the dimensionality of the vectors
        N, D = coordinate_input.size()

        christoffel_eval, _, christoffel_derv, _,_ = self._christoffel_derv(coordinate_input)
        term1 = torch.einsum('nabdg->nabgd',christoffel_derv)
        term2 = torch.einsum('nabgd->nabgd',christoffel_derv)
        term3 = torch.einsum('namg,nmbd->nabgd',christoffel_eval,christoffel_eval)
        term4 = torch.einsum('namd,nmbg->nabgd',christoffel_eval,christoffel_eval)
        return term1 - term2 + term3 - term4



class RicciTensor(nn.Module):
    '''
        Computes the Riemann tensor given by $\R^{p}_{smv}$
    '''
    def __init__(self,metric_module: SpaceTimeMetricModule):
        super(RicciTensor, self).__init__()
        self.metric = metric_module
        self.riemann = RiemannTensor(metric_module)

    def forward(self,coordinate_input: torch.Tensor):
        # input_batch has shape (N, D), where N is the batch size and D is the dimensionality of the vectors
        N, D = coordinate_input.size()
        riemann_eval =self.riemann(coordinate_input)
        return torch.einsum('nsjsk->njk',riemann_eval)

class RicciScalar(nn.Module):
    '''
        Computes the Ricci tensor given by $\R_{ij}$
    '''
    def __init__(self,metric_module: SpaceTimeMetricModule):
        super(RicciScalar, self).__init__()
        self.metric = metric_module
        self.ricci = RicciTensor(metric_module)

    def forward(self,coordinate_input: torch.Tensor):
        # input_batch has shape (N, D), where N is the batch size and D is the dimensionality of the vectors
        N, D = coordinate_input.size()
        metric_eval = self.metric(coordinate_input)
        inverse_metric = torch.linalg.inv(metric_eval)
        ricci_eval =self.ricci(coordinate_input)
        return torch.einsum('nij,nij->n',inverse_metric,ricci_eval)


class EinsteinTensor(nn.Module):
    '''
        Computes the Einstein tensor given by $\G_{ij}$
    '''
    def __init__(self,metric_module: SpaceTimeMetricModule):
        super(EinsteinTensor, self).__init__()
        self.metric = metric_module
        self.ricci = RicciTensor(metric_module)
        self.ricci_scalar = RicciScalar(metric_module)

    def forward(self,coordinate_input: torch.Tensor):
        # input_batch has shape (N, D), where N is the batch size and D is the dimensionality of the vectors
        N, D = coordinate_input.size()
        metric_eval = self.metric(coordinate_input)

        ricci_eval = self.ricci(coordinate_input)
        half_ricci_scalar = 0.5*self.ricci_scalar(coordinate_input)
        term2 = torch.einsum('n,nij->nij',half_ricci_scalar,metric_eval)
        return ricci_eval - term2
