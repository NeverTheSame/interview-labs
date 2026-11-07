merge into core.orders target
using (
    select payload:order_id::string    as order_id,
           payload:status::string      as status,
           payload:customer_id::string as customer_id
    from raw.order_events
    qualify row_number() over (partition by payload:order_id::string order by loaded_at desc) = 1
) source
on target.order_id = source.order_id
when matched then update set status = source.status, customer_id = source.customer_id
when not matched then insert (order_id, status, customer_id) values (source.order_id, source.status, source.customer_id)
